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

from cupadman import Detector, CDataset, Quaternion
import kernels
P_MIN = 1.e-6
MEM_THRESH = 0.8

class EMC():
    '''Reconstructor object using parameters from config file

    Args:
        config_file (str): Path to configuration file

    The appropriate CUDA device must be selected before initializing the class.
    Can be used with mpirun, in which case work will be divided among ranks.
    '''
    def __init__(self, config_file, num_streams=4):
        '''Parse config file and setup reconstruction

        One can just use run_iteration() after this
        '''
        # Get system properties
        self.num_streams = num_streams
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.num_proc = self.comm.size
        self.mem_size = cp.cuda.Device(cp.cuda.runtime.getDevice()).mem_info[1]

        # Parse config file
        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.num_div = config.getint('emc', 'num_div')
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.detector_file = os.path.join(os.path.dirname(config_file),
                                         config.get('emc', 'in_detector_file'))
        self.photons_file = os.path.join(os.path.dirname(config_file),
                                         config.get('emc', 'in_photons_file'))
        self.output_folder = os.path.join(os.path.dirname(config_file),
                                          config.get('emc', 'output_folder', fallback='data/'))
        self.log_file = os.path.join(os.path.dirname(config_file),
                                     config.get('emc', 'log_file', fallback='EMC.log'))
        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)

        # Setup reconstruction
        stime = time.time()
        #self.dset = Dataset(self.photons_file, self.size**2, self.need_scaling)
        # Note the following three structs have data in CPU memory
        self.det = Detector(self.detector_file)
        self.dset = CDataset(self.photons_file, self.det)
        self.quat = Quaternion(self.num_div)
        self._move_to_gpu()
        etime = time.time()

        if self.rank == 0:
            print('%d frames with %.3f photons/frame (%.3f s) (%.2f MB)' % \
                    (self.dset.num_data, self.dset.mean_count, etime-stime, self.dset.mem/1024**2))
            sys.stdout.flush()
        self.model = np.empty(3*(self.size,))
        if self.rank == 0:
            self.model[:] = np.random.random((self.size,)*3) * self.dset.mean_count / self.dset.num_pix
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)
        self.mweights = np.zeros(3*(self.size,))
        if self.need_scaling:
            self.scales = self.dset.counts / self.dset.mean_count
        else:
            self.scales = cp.ones(self.dset.num_data, dtype='f8')
        self.prob = cp.array([])

        self.bsize_model = int(np.ceil(self.det.num_pix/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))
        self.stream_list = [cp.cuda.Stream() for _ in range(self.num_streams)]

    def run_iteration(self, iternum=None):
        '''Run one iterations of EMc algorithm

        Args:
            iternum (int, optional): If specified, output is tagged with iteration number

        Current guess is assumed to be in self.model, which is updated. If scaling is included,
        the scale factors are in self.scales.
        '''

        num_rot_p = self.num_rot // self.num_proc
        if self.rank < self.num_rot % self.num_proc:
            num_rot_p += 1
        mem_frac = num_rot_p*self.dset.num_data*8/ (self.mem_size - self.dset.mem)
        num_blocks = int(np.ceil(mem_frac / MEM_THRESH))
        block_sizes = np.array([self.dset.num_data // num_blocks] * num_blocks)
        block_sizes[0:self.dset.num_data % num_blocks] += 1
        #if len(block_sizes) > 1: print(block_sizes, 'frames in each block')

        if self.prob.shape != (num_rot_p, block_sizes.max()):
            self.prob = cp.empty((num_rot_p, block_sizes.max()), dtype='f8')
        views = cp.empty((self.num_streams, self.size**2), dtype='f8')
        dmodel = cp.array(self.model)
        dmweights = cp.array(self.mweights)
        #mp = cp.get_default_memory_pool()
        #print('Mem usage: %.2f MB / %.2f MB' % (mp.total_bytes()/1024**2, self.mem_size/1024**2))

        b_start = 0
        for b in block_sizes:
            drange = (b_start, b_start + b)
            self._calculate_prob(dmodel, views, drange)
            self._normalize_prob()
            self._update_model(views, dmodel, dmweights, drange)
            b_start += b
        self._normalize_model(dmodel, dmweights, iternum)

    def _calculate_prob(self, dmodel, views, drange):
        msum = float(-self.model.sum())
        s = drange[0]
        e = drange[1]
        num_data_b = e - s
        self.bsize_data = int(np.ceil(num_data_b/32.))

        for i, r in enumerate(range(self.rank, self.num_rot, self.num_proc)):
            snum = i % self.num_streams
            self.stream_list[snum].use()
            kernels.slice_gen((self.bsize_model,), (32,),
                    (dmodel, self.quat.quats[r], 1.,
                     self.det.qvals, self.det.num_pix,
                     self.size, 1, views[snum]))
            kernels.calc_prob_all((self.bsize_data,), (32,),
                    (views[snum], num_data_b,
                     self.dset.ones[s:e], self.dset.multi[s:e],
                     self.dset.ones_accum[s:e], self.dset.multi_accum[s:e],
                     self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                     msum, self.scales[s:e], self.prob[i]))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def _normalize_prob(self):
        max_exp_p = self.prob.max(0).get()
        rmax_p = (self.prob.argmax(axis=0) * self.num_proc + self.rank).astype('i4').get()
        max_exp = np.empty_like(max_exp_p)
        self.rmax = np.empty_like(rmax_p)

        self.comm.Allreduce([max_exp_p, MPI.DOUBLE], [max_exp, MPI.DOUBLE], op=MPI.MAX)
        rmax_p[max_exp_p != max_exp] = -1
        self.comm.Allreduce([rmax_p, MPI.INT], [self.rmax, MPI.INT], op=MPI.MAX)
        max_exp = cp.array(max_exp)

        self.prob = cp.exp(cp.subtract(self.prob, max_exp, self.prob), self.prob)
        psum_p = self.prob.sum(0).get()
        psum = np.empty_like(psum_p)
        self.comm.Allreduce([psum_p, MPI.DOUBLE], [psum, MPI.DOUBLE], op=MPI.SUM)
        self.prob = cp.divide(self.prob, cp.array(psum), self.prob)
        self.prob.clip(a_min=P_MIN, out=self.prob)

    def _update_model(self, views, dmodel, dmweights, drange):
        p_norm = self.prob.sum(1)
        h_p_norm = p_norm.get()
        s = drange[0]
        e = drange[1]
        num_data_b = e - s

        dmodel[:] = 0
        dmweights[:] = 0
        for i, r in enumerate(range(self.rank, self.num_rot, self.num_proc)):
            if h_p_norm[i] == 0.:
                continue
            snum = i % self.num_streams
            self.stream_list[snum].use()
            views[snum,:] = 0
            kernels.merge_all((self.bsize_data,), (32,),
                    (self.prob[i], num_data_b,
                     self.dset.ones[s:e], self.dset.multi[s:e],
                     self.dset.ones_accum[s:e], self.dset.multi_accum[s:e],
                     self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                     views[snum]))
            views[snum] = views[snum] / p_norm[i] - self.dset.bg
            kernels.slice_merge((self.bsize_model,), (32,),
                    (views[snum], self.quat.quats[r],
                     self.det.qvals, self.det.num_pix,
                     self.size, dmodel, dmweights))
        [s.synchronize() for s in self.stream_list]
        cp.cuda.Stream().null.use()

    def _normalize_model(self, dmodel, dmweights, iternum):
        self.model = dmodel.get()
        self.mweights = dmweights.get()
        if self.rank == 0:
            self.comm.Reduce(MPI.IN_PLACE, [self.model, MPI.DOUBLE], root=0, op=MPI.SUM)
            self.comm.Reduce(MPI.IN_PLACE, [self.mweights, MPI.DOUBLE], root=0, op=MPI.SUM)
            self.model[self.mweights > 0] /= self.mweights[self.mweights > 0]

            if iternum is None:
                np.save('data/model.npy', self.model)
            else:
                np.save('data/model_%.3d.npy'%iternum, self.model)
                np.save('data/rmax_%.3d.npy'%iternum, self.rmax/self.num_rot*360.)
        else:
            self.comm.Reduce([self.model, MPI.DOUBLE], None, root=0, op=MPI.SUM)
            self.comm.Reduce([self.mweights, MPI.DOUBLE], None, root=0, op=MPI.SUM)
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)

    def _move_to_gpu(self):
        '''Move detector, dataset and quaternions to GPU'''
        self.det.qvals = cp.array(self.det.qvals)
        self.quat.quats = cp.array(self.quat.quats)

        self.dset.ones = cp.array(self.dset.ones)
        self.dset.multi = cp.array(self.dset.multi)
        self.dset.ones_accum = cp.array(self.dset.ones_accum)
        self.dset.multi_accum = cp.array(self.dset.multi_accum)
        self.dset.place_ones = cp.array(self.dset.place_ones)
        self.dset.place_multi = cp.array(self.dset.place_multi)
        self.dset.count_multi = cp.array(self.dset.count_multi)

def main():
    '''Parses command line arguments and launches EMC reconstruction'''
    import socket
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-d', '--devices', default=None,
                        help='Path to devices file')
    parser.add_argument('-s', '--streams', type=int, default=4,
                        help='Number of streams to use (default=4)')
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.rank
    num_proc = comm.size
    if args.devices is None:
        if num_proc == 1:
            print('Running on default device 0')
        else:
            print('Require a "devices" file if using multiple processes (one number per line)')
            sys.exit(1)
    else:
        with open(args.devices) as f:
            dev = int(f.readlines()[rank].strip())
            print('Rank %d: %s (Device %d)' % (rank, socket.gethostname(), dev))
            sys.stdout.flush()
            cp.cuda.Device(dev).use()

    recon = EMC(args.config_file, num_streams=args.streams)
    if rank == 0:
        print('\nIter  time(s)  change')
        sys.stdout.flush()
        avgtime = 0.
        numavg = 0
    for i in range(args.num_iter):
        m0 = cp.array(recon.model)
        stime = time.time()
        recon.run_iteration(i+1)
        etime = time.time()
        if rank == 0:
            norm = float(cp.linalg.norm(cp.array(recon.model) - m0))
            print('%-6d%-.2e %e' % (i+1, etime-stime, norm))
            sys.stdout.flush()
            if i > 0:
                avgtime += etime-stime
                numavg += 1
    if rank == 0 and numavg > 0:
        print('%.4e s/iteration on average' % (avgtime / numavg))

if __name__ == '__main__':
    main()
