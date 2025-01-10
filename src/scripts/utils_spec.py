import numpy as np
import scipy
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import tensorflow as tf
import tensorflow.keras as tfk
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
import math
import copy
import pickle

from cymetric.pointgen.pointgen_mathematica import PointGeneratorMathematica
from cymetric.pointgen.pointgen import PointGenerator
from cymetric.models.tfhelper import prepare_tf_basis, train_model
from cymetric.models.fubinistudy import FSModel

from cymetric.models.tfmodels import PhiFSModel
from cymetric.models.callbacks import SigmaCallback, VolkCallback
from cymetric.models.metrics import SigmaLoss, KaehlerLoss, TransitionLoss, VolkLoss, RicciLoss, TotalLoss



class Spectrum:
    
    
    """
    Class to compute the spectrum of the scalar Laplacian on the Calabi-Yau (CY)
    """
    
    
    
    def __init__(self, ambient, monomials, k_phi, metric_model):
        
        """
        Initializing Calabi-Yau data: dimension, ambient space, metric, monomials  
        """
        
        self.ambient = ambient
        self.monomials = monomials
        self.degrees = self.ambient + 1
        self.ncoords = int(tf.reduce_sum(self.degrees))
        self.nfold = ambient - 1
        self.k = k_phi
        self.metric_model = metric_model
        self._init_monomials(k_phi)

    def _init_monomials(self, k):
        
        """
        Monomials which give us sections of line bundles in the ambient space 
        """
        
        self.k = [k for _ in range(len(self.ambient))]
        self._generate_sections(self.k)

    def get_eigenfunction_basis(self, c_pts):
        
        """
        A basis of eigenfunctions from the sections 
        """
        
        c_pts = tf.cast(c_pts, tf.complex128)
        s_i = self.eval_sections_vec(c_pts)
        bs_j = self.eval_sections_vec(tf.math.conj(c_pts))
        sbs = tf.reshape(tf.einsum('xi,xj->xij', s_i, bs_j), (c_pts.shape[0], s_i.shape[-1]**2,))
        return tf.einsum('xa, x->xa', sbs, 1. / tf.einsum('xi,xi->x', c_pts, tf.math.conj(c_pts)) ** self.k)
        
    @tf.function
    def o_ab(self, points, weights=None, ambient=False):
        
        """
        The matrix "O" of the inner products of the eigenfunctions 
        """
        
        if ambient:
            self._generate_sections(self.k, ambient=True)
        if weights is None:
            weights = tf.ones((points.shape[0]))
        points = tf.cast(points, dtype=tf.float64)
        c_pts = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        fs = self.get_eigenfunction_basis(c_pts)
        weights = tf.cast(weights, tf.complex128)
        return 1. / fs.shape[0] * tf.einsum('xa,xb,x->ab', tf.math.conj(fs), fs, weights)

    def delta(self, points, weights=None, omegas=None, ambient=False, verbose=1):
        
        """
        Matrix representation of the Laplacian in the chosen eigenfunction basis 
        """
        
        total_num_pts = points.shape[0]
        if ambient:
            self._generate_sections(self.k, ambient=True)
        if weights is None:
            weights = tf.ones((total_num_pts,))
        # find n_chunks (max_chunk_size is 15k for GPU and 50k for CPU)
        max_chunk_size = 15000 if tf.config.list_physical_devices('GPU') else 50000
        if max_chunk_size == 15000:
            if verbose > 0: print("Batching for GPU use...")
        else:
            if verbose > 0: print("Batching for CPU use...")
        if total_num_pts <= max_chunk_size:
            n_chunks = 1
        else:
            n_chunks = -1
            for i in range(5000, max_chunk_size, 5000):
                if total_num_pts % i == 0:
                    n_chunks = total_num_pts//i
            if n_chunks == -1:
                n_chunks = total_num_pts//5000
                points = points[:n_chunks * 5000]
                weights = weights[:n_chunks * 5000]
                total_num_pts = points.shape[0]
                if verbose > 0: print("Warning, throwing away some points when batching")
                
        if verbose > 0: print("Divided {} points into {} batches with {} points each".format(total_num_pts, n_chunks, total_num_pts//n_chunks))
        batched_points = tf.reshape(points, (n_chunks, total_num_pts//n_chunks, points.shape[-1]))
        batched_weights = tf.reshape(weights, (n_chunks, total_num_pts//n_chunks))
        if omegas is not None:
            batched_omegas = tf.reshape(omegas, (n_chunks, total_num_pts//n_chunks))
        else: 
            batched_omegas = [None] * n_chunks
        if not ambient:
            return tf.reduce_sum([self.delta_cy_batched(bpts, bws, total_num_pts, bos, verbose) for bpts, bws, bos in zip(batched_points, batched_weights, batched_omegas)], axis=0)
        else:
            return tf.reduce_sum([self.delta_amb_batched(bpts, total_num_pts, verbose) for bpts in batched_points], axis=0)
    
    @tf.function
    def delta_cy_batched(self, points, weights, total_num_pts=None, omegas=None, verbose=1):
        
        """
        Pullback of the Laplacian on the ambient space to the Laplacian on the CY (for some metric model)
        """
        
        pts = tf.cast(points, dtype=tf.float64)
        if total_num_pts is None:
            total_num_pts = pts.shape[0]
        c_pts = tf.complex(pts[:, :self.ncoords], pts[:, self.ncoords:])
        bc_pts = tf.math.conj(c_pts)
        weights = tf.cast(weights, tf.complex128)
        
        if verbose > 0: print("Computing pullbacks...")
        pbs = tf.cast(self.metric_model.pullbacks(tf.cast(pts, tf.float32)), tf.complex128)  # shape [x, 1, 3]
        
        if verbose > 0: print("Computing derivatives...")
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(c_pts)
            fs = self.get_eigenfunction_basis(c_pts)
            fs1 = tf.math.real(fs)
            fs2 = tf.math.imag(fs)
        
        if verbose > 0: print("Computing inverse metrics...")
        g_inv = tf.cast(tf.linalg.inv(self.metric_model(points)), tf.complex128)  # shape [x, 1, 1]
        
        bjac1 = .5 * t1.batch_jacobian(fs1, c_pts)  # = d re(f)/ dz*
        bjac2 = .5 * t1.batch_jacobian(fs2, c_pts)  # = d im(f)/ dz*
        
        di_bfs = tf.math.conj(bjac1) - 1.j * tf.math.conj(bjac2)  # shape [x, number of fs, 3]
        di_bfs = tf.einsum('xai,xri->xar',di_bfs, pbs)
        di_bfs = tf.cast(di_bfs, tf.complex128)
        
        dbi_fs = bjac1 + 1.j * bjac2
        dbi_fs = tf.einsum('xai,xri->xar',dbi_fs, tf.math.conj(pbs))
        dbi_fs = tf.cast(dbi_fs, tf.complex128)
        
        if omegas is not None:
            g_inv = tf.reshape(tf.cast(omegas, tf.complex128), (omegas.shape[0], 1, 1))
        if verbose > 0: print("Carrying out matrix products...")
        return 2./total_num_pts * (tf.einsum('xai, xij, xbj, x->ab', di_bfs, g_inv, dbi_fs, weights))
    
    @tf.function
    def delta_amb_batched(self, points, total_num_pts=None, verbose=1):
        
        """
        Pullback of the Laplacian on the ambient space to the Laplacian on the CY (for some Fubini-Study metric (FS))
        """
        
        # this expects real points
        # first, we find the good coordinates, i.e. those that are not patch coords and that are not eliminated
        pts = tf.cast(points, dtype=tf.float64)
        if total_num_pts is None:
            total_num_pts = pts.shape[0]
        c_pts = tf.complex(pts[:, :self.ncoords], pts[:, self.ncoords:])
        bc_pts = tf.math.conj(c_pts)
        
        if verbose > 0: print("Computing pullbacks...")
        pbs = tf.cast(self.remove_patch_coord_matrix(points), tf.complex64)
        
        if verbose > 0: print("Computing derivatives...")
        # when tf computes complex derivatives, it does (don't ask why...):
        # df/dz = (del f/del z +del f/del z*)* = 2 d re(f)/ dz*
        # so we need to do the below to get the usual df/dz
        with tf.GradientTape(persistent=True) as t1:
            t1.watch(c_pts)
            fs = self.get_eigenfunction_basis(c_pts)
            fs1 = tf.math.real(fs)
            fs2 = tf.math.imag(fs)
        
        if verbose > 0: print("Computing inverse metrics...")
        g = self.metric_model._fubini_study_n_metrics(c_pts, t=fs_model.BASIS['KMODULI'][0])
        g = tf.einsum('xai,xij,xbj->xab', pbs, g, pbs)
        g_inv = tf.cast(tf.linalg.inv(g), tf.complex128)  # shape [x, 2, 2]
        
        if verbose > 0: print("Pulling back derivatives...")
        bjac1 = .5 * t1.batch_jacobian(fs1, c_pts)  # = d re(f)/ dz*
        bjac2 = .5 * t1.batch_jacobian(fs2, c_pts)  # = d im(f)/ dz*
        
        di_bfs = tf.math.conj(bjac1) - 1.j * tf.math.conj(bjac2)  # shape [x, number of fs, 3]
        di_bfs = tf.cast(di_bfs, tf.complex128)
        pbs = tf.cast(pbs, tf.complex128)
        di_bfs = tf.einsum('xai,xri->xra', pbs, di_bfs) # shape [x, number of fs, 2]
        
        dbi_fs = bjac1 + 1.j * bjac2
        dbi_fs = tf.cast(dbi_fs, tf.complex128)
        dbi_fs = tf.einsum('xai,xri->xra', pbs, dbi_fs) # shape [x, number of fs, 2]
        
        if verbose > 0: print("Carrying out matrix products...")
        return 2./total_num_pts * tf.reduce_sum(tf.einsum('xai, xji, xbj->xab', di_bfs, g_inv, dbi_fs), axis=0)
    
    def get_good_coords_mask(self, points):
        
        """
        Changing coordinates to a "good" patch in the ambient space
        """
        
        inv_one_mask = self.metric_model._get_inv_one_mask(tf.cast(points, tf.float32))
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        dQdz_indices = self.metric_model._find_max_dQ_coords(points)
        full_mask = tf.cast(inv_one_mask, dtype=tf.float32)
        for i in range(len(self.ambient)):
            dQdz_mask = -1. * tf.one_hot(dQdz_indices[:, i], self.ncoords)
            full_mask = tf.math.add(full_mask, dQdz_mask)
        n_p = tf.cast(tf.reduce_sum(tf.ones_like(full_mask[:, 0])), dtype=tf.int64)
        full_mask = tf.cast(full_mask, dtype=tf.bool)
        return full_mask
 
    def get_good_coords(self, points):
        full_mask = self.get_good_coords_mask(points)
        x_z_indices = tf.where(full_mask)
        return x_z_indices[:, 1:2]
    
    def generate_monomials(self, n, deg):
        
        """
        Generate monomials given an ambient space
        """
        
        if n == 1:
            yield (deg,)
        else:
            for i in range(deg + 1):
                for j in self.generate_monomials(n - 1, deg - i):
                    yield (i,) + j
    
    @tf.function
    def remove_patch_coord_matrix(self, points):
        # expects real points 
        inv_one_mask = self.metric_model._get_inv_one_mask(tf.cast(points, tf.float32))
        cpoints = tf.complex(points[:, :self.ncoords], points[:, self.ncoords:])
        full_mask = tf.cast(inv_one_mask, dtype=tf.float32)
        full_mask = tf.cast(full_mask, dtype=tf.bool)
        pullbacks = tf.eye(self.ambient[0] + 1, batch_shape=(points.shape[0],))
        pullbacks = tf.reshape(tf.boolean_mask(pullbacks, full_mask, axis=None), (points.shape[0],self.ambient[0], self.ambient[0] + 1))
        return pullbacks  # [x, ambient, ambient + 1]
    
    def _generate_sections(self, k, ambient=False):
        self.sections = None
        ambient_polys = [0 for i in range(len(k))]
        for i in range(len(k)):
            # create all monomials of degree k in ambient space factors
            ambient_polys[i] = list(self.generate_monomials(self.degrees[i], k[i]))
        # create all combinations for product of projective spaces
        monomial_basis = [x for x in ambient_polys[0]]
        for i in range(1, len(k)):
            lenB = len(monomial_basis)
            monomial_basis = monomial_basis*len(ambient_polys[i])
            for l in range(len(ambient_polys[i])):
                for j in range(lenB):
                    monomial_basis[l*lenB+j] = monomial_basis[l * lenB + j] + ambient_polys[i][l]
        sections = np.array(monomial_basis, dtype=np.int32)
        # reduce sections; pick (arbitrary) first monomial in point gen
        if not ambient:
            reduced = np.unique(np.where(sections - self.monomials[0] < -0.1)[0])
            sections = sections[reduced]
        self.sections = tf.cast(sections, tf.complex128)
        self.nsections = len(self.sections)

    def eval_sections_vec(self, points):
        return tf.reduce_prod(tf.math.pow(tf.expand_dims(points, 1), self.sections), axis=-1)

    def get_exact_eigenspectrum_amb(self): 
        
        """
        Get eigenspectrum for Laplacian on ambient space   
        """
        
        def get_degeneracy(m):
            return int(1./2. * (m + 1)**2 * (2*m + 2))
        eigenspec = [[4 * np.pi**2 / (np.sqrt(2) * np.pi) * k * (k + 2)] * get_degeneracy(k) for k in range(self.k[0] + 1)]
        return [item for sublist in eigenspec for item in sublist]
    
    @staticmethod
    def get_cluster_label(eig_vals, tol):
        cluster_labels = []
        curr_cluster = 0
        if len(eig_vals) == 0:
            return cluster_labels
        elif len(eig_vals) == 1:
            return [0]
        
        for i, x in enumerate(eig_vals):
            if i > 0 and abs(x - eig_vals[i-1]) > tol:
                curr_cluster += 1
            cluster_labels += [curr_cluster]
        return cluster_labels
            