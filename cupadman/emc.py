#!/usr/bin/env python

'''EMC reconstructor object and script'''

import sys
import os
import argparse
import configparser
import time

import numpy as np
import h5py
from mpi4py import MPI
import cupy as cp

from cupadman import CDetector, CDataset, Quaternion
P_MIN = 1.e-6
MEM_THRESH = 0.8

class EMC():
    '''Reconstructor object using parameters from config file

    Args:
        config_file (str): Path to configuration file

    The appropriate CUDA device must be selected before initializing the class.
    Can be used with mpirun, in which case work will be divided among ranks.
    '''
    def __init__(self, config_file, resume=False, num_streams=4):
        '''Parse config file and setup reconstruction

        One can just use run_iteration() after this
        '''
        stime = time.time()
        # Get system properties
        self.num_streams = num_streams
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.num_proc = self.comm.size
        self.mem_size = cp.cuda.Device(cp.cuda.runtime.getDevice()).mem_info[1]

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_file)
        config_dir = os.path.dirname(config_file)

        detector_fname = os.path.join(config_dir, config.get('emc', 'in_detector_file'))
        photons_fname = os.path.join(config_dir, config.get('emc', 'in_photons_file'))
        self.output_folder = os.path.join(config_dir, config.get('emc', 'output_folder', fallback='data/'))
        self.log_file = os.path.join(config_dir, config.get('emc', 'log_file', fallback='EMC.log'))
        model_fname = os.path.join(config_dir, config.get('emc', 'start_model_file', fallback=''))
        blacklist_fname = os.path.join(config_dir, config.get('emc', 'blacklist_file', fallback=''))
        scale_fname = os.path.join(config_dir, config.get('emc', 'scale_file', fallback=''))

        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)
        self.update_scale = config.getboolean('emc', 'update_scale', fallback=True)
        num_div = config.getint('emc', 'num_div', fallback=0)
        num_rot2d = config.getint('emc', 'num_rot', fallback=0)
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        beta_schedule = config.get('emc', 'beta_schedule', fallback='1. 100').split()
        beta_set = config.getfloat('emc', 'beta', fallback=-1.)
        self.beta_factor = config.getfloat('emc', 'beta_factor', fallback=1.)
        self.alpha = config.getfloat('emc', 'alpha', fallback=0.)
        sel_string = config.get('emc', 'selection', fallback='')
        rtype = config.get('emc', 'recon_type', fallback='3d')
        if rtype not in ['2d', '3d']:
            raise ValueError('Unsupported recon_type: %s'%rtype)

        # Adjust parameters if resuming reconstruction
        if resume:
            try:
                with open(self.log_file, 'r') as fptr:
                    self.last_iternum = int(fptr.readlines()[-1].split()[0])
                print('Resuming from iteration %d'%(self.last_iternum + 1))
                model_fname = self.output_folder + '/output_%.3d.h5'%self.last_iternum
                if self.need_scaling:
                    scale_fname = self.output_folder + '/output_%.3d.h5'%self.last_iternum
            except (FileNotFoundError, ValueError):
                print('WARNING: Unable to resume using %s'%self.log_file)
                self.last_iternum = 0
        else:
            self.last_iternum = 0

        # Setup reconstruction
        # -- Generate detector, dataset, quaternions
        # -- Note the following three structs have data in CPU memory
        self.det = CDetector(detector_fname, twodim=(rtype=='2d'))
        self.dset = CDataset(photons_fname, self.det)
        self.quat = Quaternion(num_div)

        if rtype == '2d':
            if num_rot2d == 0:
                raise ValueError('Need num_rot if recon_type is 2d')
            temp = np.zeros((num_rot2d, 5))
            temp[:,0] = np.arange(0, 2.*np.pi, 2.*np.pi/num_rot2d)
            temp[:,4] = 1. / num_rot2d
            self.quat.quats = temp

        self.size = int(2*np.ceil(np.linalg.norm(self.det.qvals, axis=1).max()) + 3)
        self._log_print('Calculated volume size = %ld'%self.size) ;

        self.quat.divide(self.rank, self.num_proc, self.num_modes)
        self._move_to_gpu()

        self._log_print('%d frames with %.3f photons/frame (%.3f s) (%.2f MB)' % \
                        (self.dset.num_data, self.dset.mean_count, time.time()-stime, self.dset_mem/1024**2))

        # -- Get CUDA kernels
        script_dir = os.path.dirname(os.path.abspath(__file__))
        with open(script_dir+'/kernels.cu', 'r') as f:
            kernels = cp.RawModule(code=f.read())
        self.k_slice_gen = kernels.get_function('slice_gen'+rtype)
        self.k_slice_merge = kernels.get_function('slice_merge'+rtype)
        self.k_calc_prob_all = kernels.get_function('calc_prob_all')
        self.k_merge_all = kernels.get_function('merge_all')
        self.k_norm_prob = kernels.get_function('norm_prob')
        self.k_calc_psum = cp.ReductionKernel('T prob, T beta, T max_exp', 'T psum', 'exp(beta * (prob - max_exp))', 'a + b', 'psum = a', '0', 'calc_psum2')
        self.k_calc_pnorm = cp.ReductionKernel('T prob, T scale', 'T pnorm', 'prob*scale', 'a + b', 'pnorm = a', '0', 'calc_pnorm')

        # -- Generate iterate
        # ---- Model
        ndim = 3 if rtype == '3d' else 2
        mshape = (self.num_modes,) + ndim * (self.size,)
        self.model = np.empty(mshape)
        if self.rank == 0:
            if model_fname == '':
                self._log_print('Random start')
                self.model[:] = np.random.random(mshape) * self.dset.mean_count / self.dset.num_pix
            else:
                self._log_print('Starting from %s'%model_fname)
                with h5py.File(model_fname, 'r') as fptr:
                    self.model[:] = fptr['intens'][:]
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)
        self.mweights = np.zeros(mshape)

        # ---- Scale and beta factors
        if self.need_scaling:
            self.known_scale = False
            self.dset.calc_frame_counts()
            if scale_fname == config_dir:
                #self.scales = cp.array(self.dset.fcounts) / self.dset.mean_count
                self.scales = cp.ones(self.dset.num_data)
            else:
                with h5py.File(scale_fname) as fptr:
                    self.scales = cp.array(fptr['scale'][:])
                assert self.scales.shape[0] == self.dset.num_data
                self.known_scale = True

            if beta_set >= 0.:
                self.beta_start = cp.full(self.dset.num_data, beta_set)
            else:
                self.beta_start = cp.array(np.exp(-6.5 * pow(self.dset.fcounts * 1.e-5, 0.15))) # Empirical
        else:
            self.scales = cp.ones(self.dset.num_data, dtype='f8')
            if scale_fname != config_dir:
                print('WARNING: scale_file parameter not used as need_scaling is False')

            if beta_set >= 0.:
                self.beta_start = cp.full(self.dset.num_data, beta_set)
            else:
                self.beta_start = cp.array(np.exp(-6.5 * pow(np.ones(self.dset.num_data)*self.dset.mean_count * 1.e-5, 0.15))) # Empirical
        self._log_print('Starting average beta = %f' % self.beta_start.mean())

        # -- Generate blacklist
        if blacklist_fname == config_dir:
            self.blacklist = cp.zeros(self.dset.num_data, dtype='u1')
        else:
            self.blacklist = cp.array(np.loadtxt(blacklist_fname, dtype='u1'))
            assert self.blacklist.shape[0] == self.dset.num_data
        if sel_string == 'odd_only':
            self.blacklist[np.where(self.blacklist==0)[0][::2]] = 1
        elif sel_string == 'even_only':
            self.blacklist[np.where(self.blacklist==0)[0][1::2]] = 1
        elif sel_string != '':
            raise ValueError('Selection string must be odd_only or even_only')
        self.num_blacklist = self.blacklist.sum()
        self._log_print('%d/%d blacklisted frames'%(self.num_blacklist, self.dset.num_data))

        # -- Initialize other attributes
        self.beta_jump = float(beta_schedule[0])
        self.beta_period = int(beta_schedule[1])
        self.prob = cp.array([])
        self.bsize_pixel = int(np.ceil(self.det.num_pix/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]
        self.likelihood = np.zeros(self.dset.num_data)
        self.mutual_info = np.zeros(self.dset.num_data)
        rotindarr = cp.arange(self.rank, self.quat.num_rot*self.num_modes, self.num_proc) // self.num_modes
        self.qweight = self.quats[rotindarr, 4] / self.num_modes

        self._log_print('Completed setup: %f s' % (time.time() - stime), all_ranks=True)

    def run_iteration(self, iternum=1):
        '''Run one iterations of EMC algorithm

        Args:
            iternum (int): Iteration number

        Current guess is assumed to be in self.model, which is updated. If scaling is included,
        the scale factors are in self.scales.
        '''
        stime = time.time()
        mem_frac = self.quat.num_rot_p*self.dset.num_data*8/ (self.mem_size - self.dset_mem)
        num_blocks = int(np.ceil(mem_frac / MEM_THRESH))
        block_sizes = np.array([self.dset.num_data // num_blocks] * num_blocks)
        block_sizes[0:self.dset.num_data % num_blocks] += 1
        if len(block_sizes) > 1:
            block_str = '%d' % block_sizes.min()
            if blocks_sizes.max() != blocks_sizes.min():
                block_str += '-%d' % block_sizes.max()
            self._log_print('\t%d blocks with %s frames in each block' % (len(block_sizes), block_str))

        if self.prob.shape != (self.quat.num_rot_p, block_sizes.max()):
            self.prob = cp.empty((self.quat.num_rot_p, block_sizes.max()), dtype='f8')
        views = cp.empty((self.num_streams, self.det.num_pix), dtype='f8')
        dmodel = cp.array(self.model)
        dmweights = cp.array(self.mweights)
        curr_factor = self.beta_jump ** ((iternum-1) // self.beta_period) * self.beta_factor
        self.beta = self.beta_start * curr_factor
        cp.clip(self.beta, 0., 1., self.beta)
        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))
        self._calculate_rescale(dmodel, views)
        if iternum > 1 and self.need_scaling:
            self.known_scale = True

        b_start = 0
        for i, b in enumerate(block_sizes):
            drange = (b_start, b_start + b)
            if b < block_sizes.max():
                self.prob = cp.resize(self.prob, (self.quat.num_rot_p, b))
            self._calculate_prob(dmodel, views, drange)
            self._normalize_prob(drange)
            self._update_model(views, dmodel, dmweights, drange)
            b_start += b
            if num_blocks > 1:
                self._log_print('\tFinished block %d/%d'%(i+1, num_blocks))
        diff = self._normalize_model(dmodel, dmweights, iternum)
        etime = time.time()
        self._log_print('Completed maximize: %f s' % (etime-stime))
        if self.rank == 0:
            self._write_log_file(iternum, etime-stime, diff)

    def _calculate_rescale(self, dmodel, views):
        stime = time.time()
        self.vsum = cp.zeros(self.quat.num_rot_p, dtype='f8')
        total = 0.

        for i, r in enumerate(range(self.rank, self.quat.num_rot*self.num_modes, self.num_proc)):
            snum = i % self.num_streams
            modenum = r % self.num_modes
            rotind = r // self.num_modes

            self.stream_list[snum].use()
            self.k_slice_gen((self.bsize_pixel,), (32,),
                    (dmodel[modenum], self.quats[rotind], self.pixvals,
                     self.dmask, 0., self.det.num_pix,
                     self.size, views[snum]))
            self.vsum[i] = views[snum][self.det.raw_mask==0].sum()
            total += self.vsum[i] * self.quat.quats[rotind,4]
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

        total = self.comm.allreduce(float(total), op=MPI.SUM)
        self.rescale = self.dset.mean_count / total * self.num_modes
        self._log_print('\trescale\t%f (= %e)'%(time.time() - stime, self.rescale))

    def _calculate_prob(self, dmodel, views, drange):
        stime = time.time()
        s = drange[0]
        e = drange[1]
        num_data_b = e - s
        self.bsize_data = int(np.ceil(num_data_b/32.))

        for i, r in enumerate(range(self.rank, self.quat.num_rot*self.num_modes, self.num_proc)):
            snum = i % self.num_streams
            modenum = r % self.num_modes
            rotind = r // self.num_modes

            self.stream_list[snum].use()
            self.k_slice_gen((self.bsize_pixel,), (32,),
                    (dmodel[modenum], self.quats[rotind], self.pixvals,
                     self.dmask, 1., self.det.num_pix,
                     self.size, views[snum]))
            if self.need_scaling and self.known_scale:
                initvals = cp.log(self.quats[rotind,4]) - self.vsum[i] * self.scales[s:e]
            else:
                initvals = cp.log(self.quats[rotind,4]) - self.vsum[i] * cp.full((e-s,), self.rescale)
            self.k_calc_prob_all((self.bsize_data,), (32,),
                    (views[snum], num_data_b, self.blacklist[s:e],
                     self.ones[s:e], self.multi[s:e],
                     self.ones_accum[s:e], self.multi_accum[s:e],
                     self.place_ones, self.place_multi, self.count_multi,
                     self.dmask, initvals, self.prob[i]))
            if (r % (self.quat.num_rot * self.num_modes // 10) == 0):
                self._log_print('\t\tFinished r = %d/%d'%(r, self.quat.num_rot*self.num_modes), all_ranks=True)
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()
        self._log_print('\tprob\t%f'%(time.time() - stime))

    def _normalize_prob(self, drange):
        stime = time.time()
        s = drange[0]
        e = drange[1]
        num_data_b = int(e - s)

        # Calculate max_exp
        max_exp_p = self.prob.max(0).get()
        rmax_p = (self.prob.argmax(axis=0) * self.num_proc + self.rank).astype('i4').get()
        max_exp = np.empty_like(max_exp_p)
        self.rmax = np.empty_like(rmax_p)

        # Reduce max_exp across ranks
        self.comm.Allreduce([max_exp_p, MPI.DOUBLE], [max_exp, MPI.DOUBLE], op=MPI.MAX)
        rmax_p[max_exp_p != max_exp] = -1
        self.comm.Allreduce([rmax_p, MPI.INT], [self.rmax, MPI.INT], op=MPI.MAX)
        max_exp = cp.array(max_exp)

        # Calculate normalization factor over r
        psum_p = self.k_calc_psum(self.prob, self.beta[s:e], max_exp, axis=0)
        psum = np.empty(psum_p.shape)
        self.comm.Allreduce([psum_p.get(), MPI.DOUBLE], [psum, MPI.DOUBLE], op=MPI.SUM)
        psum = cp.array(psum)

        # Get normalized prob and other metrics
        s_norm_p = cp.empty(num_data_b)
        like_p = cp.empty(num_data_b)
        info_p = cp.empty(num_data_b)

        self.k_norm_prob((self.bsize_data,), (32,),
                (self.beta[s:e], max_exp, psum, self.qweight, self.vsum,
                 self.blacklist[s:e], self.quat.num_rot_p, num_data_b,
                 self.prob, s_norm_p, like_p, info_p))

        self.likelihood[s:e] = 0.
        self.mutual_info[s:e] = 0.
        self.s_norm = np.zeros(s_norm_p.shape)
        self.comm.Allreduce([s_norm_p.get(), MPI.DOUBLE], [self.s_norm, MPI.DOUBLE], op=MPI.SUM)
        self.comm.Allreduce([info_p.get(), MPI.DOUBLE], [self.mutual_info[s:e], MPI.DOUBLE], op=MPI.SUM)
        self.comm.Allreduce([like_p.get(), MPI.DOUBLE], [self.likelihood[s:e], MPI.DOUBLE], op=MPI.SUM)

        self._log_print('\tnorm\t%f'%(time.time() - stime))

    def _update_model(self, views, dmodel, dmweights, drange):
        stime = time.time()
        s = drange[0]
        e = drange[1]
        num_data_b = e - s

        if self.need_scaling:
            p_norm = self.k_calc_pnorm(self.prob, self.scales[s:e], axis=1)
            if self.update_scale:
                self.scales[s:e] = cp.array(self.dset.fcounts[s:e] / self.s_norm) / self.rescale
        else:
            p_norm = self.prob.sum(1)

        h_p_norm = p_norm.get()

        dmodel[:] = 0
        dmweights[:] = 0
        for i, r in enumerate(range(self.rank, self.quat.num_rot*self.num_modes, self.num_proc)):
            if h_p_norm[i] == 0.:
                continue
            snum = i % self.num_streams
            modenum = r % self.num_modes
            rotind = r // self.num_modes

            self.stream_list[snum].use()
            views[snum,:] = 0
            self.k_merge_all((self.bsize_data,), (32,),
                    (self.prob[i], num_data_b, self.blacklist[s:e],
                     self.ones[s:e], self.multi[s:e],
                     self.ones_accum[s:e], self.multi_accum[s:e],
                     self.place_ones, self.place_multi, self.count_multi,
                     self.dmask, views[snum]))
            #views[snum] = views[snum] / p_norm[i] - self.dset.bg
            views[snum] = views[snum] / p_norm[i]
            self.k_slice_merge((self.bsize_pixel,), (32,),
                    (views[snum], self.quats[rotind],
                     self.pixvals, self.dmask, self.det.num_pix,
                     self.size, dmodel[modenum], dmweights[modenum]))
            if (r % (self.quat.num_rot*self.num_modes // 10) == 0):
                self._log_print('\t\tFinished r = %d/%d'%(r, self.quat.num_rot*self.num_modes), all_ranks=True)
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()
        self._log_print('\tupdate\t%f'%(time.time() - stime))

    def _normalize_model(self, dmodel, dmweights, iternum):
        stime = time.time()
        old_model = np.copy(self.model)
        self.model = dmodel.get()
        self.mweights = dmweights.get()
        if self.rank == 0:
            self.comm.Reduce(MPI.IN_PLACE, [self.model, MPI.DOUBLE], root=0, op=MPI.SUM)
            self.comm.Reduce(MPI.IN_PLACE, [self.mweights, MPI.DOUBLE], root=0, op=MPI.SUM)
            self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]
            self._sym_friedel()

            if self.need_scaling and self.model.ndim == 4:
                mscale = float(self.scales.sum()) / float(self.dset.num_data - self.num_blacklist)
                self.scales /= mscale
                self.model *= mscale

            self._save_output(iternum)
            diff = np.linalg.norm((self.model - old_model).ravel()) / self.model.size**0.5
            if self.alpha > 0.:
                self.model = old_model * self.alpha + self.model * (1. - self.alpha)
        else:
            self.comm.Reduce([self.model, MPI.DOUBLE], None, root=0, op=MPI.SUM)
            self.comm.Reduce([self.mweights, MPI.DOUBLE], None, root=0, op=MPI.SUM)
            diff = 0.
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)
        self._log_print('\tsync\t%f'%(time.time() - stime))
        return diff

    def _save_output(self, iternum):
        fptr = h5py.File(self.output_folder + '/output_%.3d.h5'%iternum, 'w')
        fptr['intens'] = self.model
        fptr['inter_weight'] = self.mweights
        fptr['orientations'] = self.rmax
        fptr['scale'] = self.scales.get()
        fptr['mutual_info'] = self.mutual_info
        if self.num_modes > 1:
            fptr['occupancies'] = self.prob.reshape(self.quat.num_rot, self.num_modes, -1).sum(0).T.get()
            #fptr['occupancies'] = self.prob.reshape(self.num_modes, self.quat.num_rot, -1).sum(1).T.get()
        fptr.close()

    def _write_log_file(self, iternum, itertime, norm):
        if iternum == 1:
            if os.path.exists(self.log_file):
                i = 1
                dirname = os.path.dirname(self.log_file)
                basename = os.path.basename(self.log_file)
                while True:
                    bak_fname = os.path.join(dirname, '.%s.bak%d'%(basename, i))
                    if os.path.exists(bak_fname):
                        i += 1
                    else:
                        break
                self._log_print('Copying old log file to %s'%bak_fname)
                os.system('cp %s %s' % (self.log_file, bak_fname))
            fptr = open(self.log_file, 'w')
            fptr.write('Cryptotomography with the EMC algorithm using MPI+CUDA\n\n')
            if self.num_blacklist > 0:
                fptr.write('Data parameters:\n\tnum_data = %d/%d\n\tmean_count = %f\n\n' % (self.dset.num_data-self.num_blacklist, self.dset.num_data, self.dset.mean_count))
            else:
                fptr.write('Data parameters:\n\tnum_data = %d\n\tmean_count = %f\n\n' % (self.dset.num_data, self.dset.mean_count))
            if len(self.model.shape) == 4:
                fptr.write('System size:\n\tnum_rot = %d\n\tnum_pix = %d/%d\n\tsystem_volume = %d X %d X %d X %d\n\n' % (self.quat.num_rot, (self.det.raw_mask==0).sum(), self.det.num_pix, self.num_modes, self.size, self.size ,self.size))
            else:
                fptr.write('System size:\n\tnum_rot = %d\n\tnum_pix = %d/%d\n\tsystem_volume = %d X %d X %d\n\n' % (self.quat.num_rot, (self.det.raw_mask==0).sum(), self.det.num_pix, self.num_modes, self.size ,self.size))
            fptr.write('Reconstruction parameters:\n\t')
            fptr.write('num_proc = %d\n\t'%self.num_proc)
            fptr.write('alpha = %f\n\t'%self.alpha)
            fptr.write('beta = %f\n\t'%self.beta_start.mean())
            fptr.write('need_scaling = %s\n\n'%('yes' if self.need_scaling else 'no'))
            fptr.write('Iter  time   rms_change   info_rate  log-likelihood  num_rot  beta\n')
        else:
            fptr = open(self.log_file, 'a')

        mean_like = self.likelihood[self.blacklist.get()==0].mean()
        mean_info = self.mutual_info[self.blacklist.get()==0].mean()
        fptr.write('%-6d%-.2f   %.4e   %f   %e    %-8d %f\n' % (iternum, itertime, norm, mean_info, mean_like, self.quat.num_rot, self.beta.mean()))
        fptr.close()

    def _move_to_gpu(self):
        '''Move detector, dataset and quaternions to GPU'''
        self.dmask = cp.array(self.det.raw_mask)
        self.pixvals = cp.array(np.concatenate((self.det.qvals, self.det.corr[:,np.newaxis]), axis=1).ravel())

        self.quats = cp.array(self.quat.quats)

        init_mem = cp.get_default_memory_pool().used_bytes()
        self.ones = cp.array(self.dset.ones)
        self.multi = cp.array(self.dset.multi)
        self.ones_accum = cp.array(self.dset.ones_accum)
        self.multi_accum = cp.array(self.dset.multi_accum)
        self.place_ones = cp.array(self.dset.place_ones)
        self.place_multi = cp.array(self.dset.place_multi)
        self.count_multi = cp.array(self.dset.count_multi)
        self.dset_mem = cp.get_default_memory_pool().used_bytes() - init_mem

    def _log_print(self, string, all_ranks=False):
        if all_ranks or self.rank == 0:
            sys.stderr.write(string+'\n')
            sys.stderr.flush()

    def _sym_friedel(self):
        if self.model.ndim == 3: # 2D model
            self.model = 0.5 * (self.model*self.mweights + self.model[:,::-1,::-1]*self.mweights[:,::-1,::-1])
            self.mweights = 0.5 * (self.mweights + self.mweights[:,::-1,::-1])
            self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]
        elif self.model.ndim == 4: # 3D model
            self.model = 0.5 * (self.model*self.mweights + self.model[:,::-1,::-1,::-1]*self.mweights[:,::-1,::-1,::-1])
            self.mweights = 0.5 * (self.mweights + self.mweights[:,::-1,::-1,::-1])
            self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    import socket
    parser = argparse.ArgumentParser(description='Cryptotomography with the EMC algorithm using MPI+CUDA')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-r', '--resume', action='store_true',
                        help='Resume reconstruction')
    parser.add_argument('-d', '--devices', default=None,
                        help='Comma-separated list of device numbers')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_proc = comm.size
    if args.devices is None:
        if num_proc == 1:
            print('Running on default device 0')
            sys.stdout.flush()
        else:
            print('Require "devices" option when using multiple processes')
            sys.exit(1)
    else:
        dev = args.devices.split(',')
        if len(dev) != num_proc:
            print('Number of devices (with repetition) must equal number of MPI ranks')
            sys.exit(1)
        cp.cuda.Device(int(dev[rank])).use()

    recon = EMC(args.config_file, resume=args.resume, num_streams=args.streams)
    for i in range(args.num_iter):
        recon.run_iteration(recon.last_iternum + i + 1)
    recon._log_print('Finished all iterations')

if __name__ == '__main__':
    main()
