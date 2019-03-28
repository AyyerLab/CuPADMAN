#!/usr/bin/env python

import sys
import os
import argparse
import configparser
import time

import numpy as np
import h5py
from mpi4py import MPI
import cupy as cp
from cupyx.scipy import ndimage
import cupyx
#cp.cuda.Device(2).use()
P_MIN = 1.e-6

import kernels

class Dataset():
    def __init__(self, photons_file, num_pix, need_scaling=False):
        self.photons_file = photons_file
        self.num_pix = num_pix

        with h5py.File(self.photons_file, 'r') as fptr:
            if self.num_pix != fptr['num_pix'][0]:
                raise AttributeError('Number of pixels in photons file does not match')
            self.num_data = fptr['place_ones'].shape[0]
            try:
                self.ones = cp.array(fptr['ones'][:])
            except KeyError:
                self.ones = cp.array([len(fptr['place_ones'][i]) for i in range(self.num_data)]).astype('i4')
            self.ones_accum = cp.roll(self.ones.cumsum(), 1)
            self.ones_accum[0] = 0
            self.place_ones = cp.array(np.hstack(fptr['place_ones'][:]))

            try:
                self.multi = cp.array(fptr['multi'][:])
            except KeyError:
                self.multi = cp.array([len(fptr['place_multi'][i]) for i in range(self.num_data)]).astype('i4')
            self.multi_accum = cp.roll(self.multi.cumsum(), 1)
            self.multi_accum[0] = 0
            self.place_multi = cp.array(np.hstack(fptr['place_multi'][:]))
            self.count_multi = np.hstack(fptr['count_multi'][:])

            self.mean_count = float((self.place_ones.shape[0] + self.count_multi.sum()) / self.num_data)
            if need_scaling:
                self.counts = self.ones + cp.array([self.count_multi[m_a:m_a+m].sum() for m, m_a in zip(self.multi.get(), self.multi_accum.get())])
            self.count_multi = cp.array(self.count_multi)

class EMC():
    def __init__(self, config_file):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.rank
        self.num_proc = self.comm.size

        config = configparser.ConfigParser()
        config.read(config_file)

        self.size = config.getint('parameters', 'size')
        self.num_rot = config.getint('emc', 'num_rot')
        self.num_modes = config.getint('emc', 'num_modes', fallback=1)
        self.photons_file = os.path.join(os.path.dirname(config_file),
                                         config.get('emc', 'in_photons_file'))
        self.output_folder = os.path.join(os.path.dirname(config_file),
                                          config.get('emc', 'output_folder', fallback='data/'))
        self.log_file = os.path.join(os.path.dirname(config_file),
                                     config.get('emc', 'log_file', fallback='EMC.log'))
        self.need_scaling = config.getboolean('emc', 'need_scaling', fallback=False)

        stime = time.time()
        self.dset = Dataset(self.photons_file, self.size**2, self.need_scaling)
        etime = time.time()
        if self.rank == 0:
            print('%d frames with %.3f photons/frame (%f s)' % (self.dset.num_data, self.dset.mean_count, etime-stime))
            sys.stdout.flush()
        self.model = np.empty((self.size, self.size))
        if self.rank == 0:
            self.model[:] = np.random.random((self.size, self.size)) * self.dset.mean_count / self.dset.num_pix
        self.comm.Bcast([self.model, MPI.DOUBLE], root=0)
        self.mweights = np.zeros((self.size, self.size), dtype='f8')
        if self.need_scaling:
            self.scales = self.dset.counts / self.dset.mean_count
        else:
            self.scales = cp.ones(self.dset.num_data, dtype='f8')

        self.bsize_model = int(np.ceil(self.size/32.))
        self.bsize_data = int(np.ceil(self.dset.num_data/32.))

    def run_iteration(self, iternum=None):
        num_rot_p = self.num_rot // self.num_proc
        if self.rank < self.num_rot % self.num_proc:
            num_rot_p += 1
        self.prob = cp.empty((num_rot_p, self.dset.num_data), dtype='f8')
        view = cp.empty(self.size**2, dtype='f8')
        dmodel = cp.array(self.model)
        dmweights = cp.array(self.mweights)

        self._calculate_prob(dmodel, view)                  # CUDA stuff
        self._normalize_prob()                              # MPI stuff
        self._update_model(view, dmodel, dmweights)         # CUDA stuff
        self._normalize_model(dmodel, dmweights, iternum)   # MPI stuff

    def _calculate_prob(self, dmodel, view):
        msum = float(-self.model.sum())

        for i, r in enumerate(range(self.rank, self.num_rot, self.num_proc)):
            kernels.slice_gen((self.bsize_model,)*2, (32,)*2,
                (dmodel, r/self.num_rot*2.*np.pi, 1.,
                 self.size, 1, view))
            kernels.calc_prob_all((self.bsize_data,), (32,),
                (view, self.dset.num_data,
                 self.dset.ones, self.dset.multi,
                 self.dset.ones_accum, self.dset.multi_accum,
                 self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                 msum, self.scales, self.prob[i]))

    def _normalize_prob(self):
        max_exp_p = self.prob.max(0).get()
        rmax_p = (self.prob.argmax(axis=0) * self.num_proc + self.rank).astype('i4').get()
        max_exp = np.empty_like(max_exp_p)
        self.rmax = np.empty_like(rmax_p)

        self.comm.Allreduce([max_exp_p, MPI.DOUBLE], [max_exp, MPI.DOUBLE], op=MPI.MAX)
        rmax_p[max_exp_p != max_exp] = -1
        self.comm.Allreduce([rmax_p, MPI.INT], [self.rmax, MPI.INT], op=MPI.MAX)
        max_exp = cp.array(max_exp)

        self.prob = cp.exp(self.prob - max_exp)
        psum_p = self.prob.sum(0).get()
        psum = np.empty_like(psum_p)
        self.comm.Allreduce([psum_p, MPI.DOUBLE], [psum, MPI.DOUBLE], op=MPI.SUM)
        self.prob /= cp.array(psum)
        self.prob[self.prob < P_MIN] = 0

    def _update_model(self, view, dmodel, dmweights):
        p_norm = self.prob.sum(1)
        h_p_norm = p_norm.get()

        dmodel[:] = 0
        dmweights[:] = 0
        for i, r in enumerate(range(self.rank, self.num_rot, self.num_proc)):
            if h_p_norm[i] == 0.:
                continue
            view[:] = 0
            kernels.merge_all((self.bsize_data,), (32,),
                (self.prob[i], self.dset.num_data,
                 self.dset.ones, self.dset.multi,
                 self.dset.ones_accum, self.dset.multi_accum,
                 self.dset.place_ones, self.dset.place_multi, self.dset.count_multi,
                 view))
            kernels.slice_merge((self.bsize_model,)*2, (32,)*2,
                (view/p_norm[i], r/self.num_rot*2.*np.pi,
                 self.size, dmodel, dmweights))

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

def main():
    import socket
    parser = argparse.ArgumentParser(description='In-plane rotation EMC')
    parser.add_argument('num_iter', type=int,
                        help='Number of iterations')
    parser.add_argument('-c', '--config_file', default='config.ini',
                        help='Path to configuration file (default: config.ini)')
    parser.add_argument('-d', '--devices', default=None,
                        help='Path to devices file')
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

    recon = EMC(args.config_file)
    if rank == 0:
        print('\nIter  time     change')
        sys.stdout.flush()
    for i in range(args.num_iter):
        m0 = cp.array(recon.model)
        stime = time.time()
        recon.run_iteration(i+1)
        etime = time.time()
        if rank == 0:
            norm = float(cp.linalg.norm(cp.array(recon.model) - m0))
            print('%-6d%-.2e %e' % (i+1, etime-stime, norm))
            sys.stdout.flush()

if __name__ == '__main__':
    main()
