import MDAnalysis as mda
import multiprocess as mp
import numpy as np
import time
from density_field import density_field
import sys
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


# Import simulation data
temp = int(sys.argv[1])
phi = float(sys.argv[2])
rep = int(sys.argv[3])
print("Processing: T = {} K, Phi = {:.1f} kJ/mol, Rep = {}".format(temp, phi, rep))
print("Loading Simulation and CHILL+ Data")
trj_path = './trajectories/{}K/{:.1f}/{}K_{:.1f}_REP{}'.format(temp, phi, temp, phi, rep)
xtc = trj_path + '/{}K_{:.1f}_REP{}_whole.xtc'.format(temp, phi, rep)
gro = trj_path + '/{}K_{:.1f}_REP{}_npt.gro'.format(temp, phi, rep)
iso_path = './isosurface/{}K/{:.1f}/{}K_{:.1f}_REP{}/'.format(temp, phi, temp, phi, rep)

# Set plot parameters
plt.clf()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
color_arr = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'grey', 'pink', 'tan', 'cyan', 'black']


# Load CHILL+ data
def load_index(path):
    idx = []
    with open(path) as f:
        for line in f:
            idx.append([int(x) for x in line.split()[1:]])

    return idx


chillplus_path = './chillplus/{}K/{:.1f}/{}K_{:.1f}_REP{}/'.format(temp, phi, temp, phi, rep)
index_files = ['Hex.index', 'IntIce.index', 'Cubic.index',
               'SurfIce.index', 'Liq.index']
idx_hex = load_index(chillplus_path + index_files[0])
idx_int = load_index(chillplus_path + index_files[1])
idx_cubic = load_index(chillplus_path + index_files[2])
idx_surf = load_index(chillplus_path + index_files[3])
idx_liq = load_index(chillplus_path + index_files[4])


# Load trajectory data
u = mda.Universe(gro, xtc)
ow = u.select_atoms('name OW')
oi = u.select_atoms('name OI')
frames = u.trajectory
n_frames = len(frames)
frames_arr = np.linspace(0, n_frames - 1, n_frames, dtype=int)


# Get indices
ow_idx = (ow.indices + 1).astype(np.int32)
oi_idx = (oi.indices + 1).astype(np.int32)


# Initialize box vector array
box_vector = np.zeros((n_frames, 6), dtype='float32')


# Merge all ice data
chillplus_ice = []
for i in range(n_frames):
    chillplus_ice.append(np.sort(np.array(idx_hex[i] + idx_int[i] + idx_cubic[i] + idx_surf[i], dtype='int32')))
chillplus_ice = np.array(chillplus_ice, dtype=object)


mobile_ice_pi_pos = []  # Store positions for mobile ice and pseudo-ice (For field fitting)
mobile_ice_liq_pos = []  # Store the positions for mobile ice and Liq near ice (For true lambda)
mobile_ice_liq_idx = []  # Store indices for mobile ice and Liq near ice (For true lambda)
lambda_chillplus = []
lambda_chillplus = []
oi_max_z = np.zeros(n_frames, dtype=np.float32)
# Collect trajectory data
for i, ts in enumerate(frames):
    # Load box dimensions
    box_vector[i] = frames.ts.dimensions

    # Load OW and OI positions
    ow_pos = ow.positions
    oi_pos = oi.positions
    oi_max_z[i] = np.max(oi_pos[:, 2])

    # Combine mobile ice with pseudo-ice for field fitting
    mobile_ice_cond = np.in1d(ow_idx, chillplus_ice[i])
    mobile_ice_pos = ow_pos[mobile_ice_cond]
    mobile_ice_pi_pos.append(np.r_[oi_pos, mobile_ice_pos])
    mobile_ice_idx = np.intersect1d(ow_idx, chillplus_ice[i])
    lambda_chillplus.append(mobile_ice_idx.shape[0])

    # CHILL+ Liq that are also type OW
    ow_liq_idx = np.intersect1d(ow_idx, idx_liq[i]).astype(np.int32)
    ow_liq_cond = np.in1d(ow_idx, idx_liq[i])
    ow_liq_pos = ow_pos[ow_liq_cond]

    # Find CHILL+ Liq that are near CHILL+ Ice
    liq_near_ice = mda.lib.nsgrid.FastNS(3.5, mobile_ice_pos, box_vector[i], pbc=True)
    liq_near_ice_results = liq_near_ice.search(ow_liq_pos)
    liq_near_ice_pair_idx = np.unique(liq_near_ice_results.get_pairs()[:, 0])  # For smart indexing
    liq_near_ice_idx = ow_liq_idx[liq_near_ice_pair_idx]
    liq_near_ice_pos = ow_liq_pos[liq_near_ice_pair_idx]

    # Combine the mobile ice and Liq near ice for true lambda searching
    mobile_ice_liq_idx_sort = np.argsort(np.r_[mobile_ice_idx, liq_near_ice_idx])
    mobile_ice_liq_idx.append(np.sort(np.r_[mobile_ice_idx, liq_near_ice_idx]))
    mobile_ice_liq_pos.append(np.r_[mobile_ice_pos, liq_near_ice_pos][mobile_ice_liq_idx_sort])

# Convert to numpy arrays
mobile_ice_pi_pos = np.array(mobile_ice_pi_pos, dtype=object)
mobile_ice_liq_pos = np.array(mobile_ice_liq_pos, dtype=object)
mobile_ice_liq_idx = np.array(mobile_ice_liq_idx, dtype=object)
lambda_chillplus = np.array(lambda_chillplus)


# Function for multiprocessing
def run(frame):
    df = density_field(mobile_ice_pi_pos[frame], box_vector[frame], np.array([80, 80, 80]))
    field = df.density_field_grid_search()

    verts, faces = df.iso_3d_mcubes(field)

    verts_new, faces_new, verts_def, faces_def = df.split_mcubes_surf(verts, faces, defects=True)

    surface = df.iso_3d_grid(verts_new[0], n_grid=np.array([100, 100, 100]), type='linear')

    area = df.area(verts_new[0], faces_new[0])
    h = np.copy(verts_new[0])
    h[:, 2] = h[:, 2] - oi_max_z[frame]
    vol = df.volume(h[faces_new[0]])
    vol_def, dist_def = df.defect_vol(verts_new[0][faces_new[0]], verts_def, faces_def)

    # Calculate true lambda with NN and MT method
    n_true_nn, true_idx_nn, false_idx_nn, false_dist_nn = df.true_ice_nn(mobile_ice_liq_pos[frame], verts_new[0], verts_new[-1])

    n_true_mt, true_idx_mt, false_idx_mt, false_dist_mt = df.true_ice_mt(mobile_ice_liq_pos[frame], verts_new[0][faces_new[1]], verts_new[-1][faces_new[-1]])

    return surface, verts_new, faces_new, area, vol, vol_def, dist_def, n_true_nn, true_idx_nn, false_idx_nn, false_dist_nn, n_true_mt, true_idx_mt, false_idx_mt, false_dist_mt


# Begin isosurface calculation
try:
    n_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
except KeyError:
    print("SLURM CPUs not found, using CPUs detectable by multiprocessing")
    n_cpus = mp.cpu_count()

print("Begin Isosurface Calculation Using {} SLURM CPUs".format(n_cpus))
start = time.time()
pool = mp.Pool(n_cpus)
surface, verts_new, faces_new, area, vol, vol_def, dist_def, n_true_nn, true_idx_nn, false_idx_nn, false_dist_nn, n_true_mt, true_idx_mt, false_idx_mt, false_dist_mt = zip(*pool.imap(run, list(frames_arr)))
pool.close()
pool.join()
end = time.time()
print("ELAPSED TIME: {:.2f}".format(end - start))

n_true_nn = np.array(n_true_nn)
n_true_mt = np.array(n_true_mt)


# Write data to file
print("Writing True Ice Data to File")
f = open(iso_path + 'true_ice.dat', 'w+')
f.write("# t [ps] \t lambda_true_mt \t false_neg_mt \t lambda_true_nn \t false_neg_nn \t Area [A^2]")
f.write(" \t Volume [A^3] \t Interface Avg. Loc. [A] \t Interface Std. Loc. [A]\n")
for i in frames_arr:
    false_neg_nn = np.intersect1d(idx_liq[i], mobile_ice_liq_idx[i][true_idx_nn[i]]).shape[0]
    false_neg_mt = np.intersect1d(idx_liq[i], mobile_ice_liq_idx[i][true_idx_mt[i]]).shape[0]
    f.write("{} \t {} \t {} ".format(i * 10, n_true_mt[i], false_neg_mt))
    f.write("\t {} \t {} \t {:.2f} \t {:.2f} ".format(n_true_nn[i], false_neg_nn, area[i], vol[i]))
    f.write("\t {:.5f} \t {:.5f}\n".format(np.average(verts_new[i][0][:, 2]), np.std(verts_new[i][0][:, 2])))
f.close()


# Write false positive data to file
print("Writing False Positive Data to File")
def false_positives_out(data, fname):
    f = open(iso_path + '{}.dat'.format(fname), 'w+')
    f.write("# Number of False Positives \t Hex \t Cubic \t Int \t Surf\n")
    for i in frames_arr:
        n_false = mobile_ice_liq_idx[i][data[i]]
        n_false_hex = np.intersect1d(n_false, idx_hex[i]).shape[0]
        n_false_cubic = np.intersect1d(n_false, idx_cubic[i]).shape[0]
        n_false_int = np.intersect1d(n_false, idx_int[i]).shape[0]
        n_false_surf = np.intersect1d(n_false, idx_surf[i]).shape[0]
        n_false_liq = np.intersect1d(n_false, idx_liq[i]).shape[0]
        f.write("{} \t {} \t {} \t ".format(n_false.shape[0] - n_false_liq, n_false_hex, n_false_cubic))
        f.write("{} \t {}\n".format(n_false_int, n_false_surf))
    f.close()


false_positives_out(false_idx_mt, 'false_positives_mt')
false_positives_out(false_idx_nn, 'false_positives_nn')


# Write defects to file
print("Writing Defect Data to File")
f = open(iso_path + 'defect_volume_{}K_{:.1f}_REP{}.dat'.format(temp, phi, rep), 'w+')
f.write("# t [ps] \t Defect Volume [A^3]\n")
for i, v in enumerate(vol_def):
    f.write('{} \t '.format(i * 10))
    for j in range(len(v)):
        if j != len(v) - 1:
            f.write('{} \t'.format(v[j]))
        else:
            f.write('{}\n'.format(v[j]))
f.close()

f = open(iso_path + 'defect_distance_{}K_{:.1f}_REP{}.dat'.format(temp, phi, rep), 'w+')
f.write("# t [ps] \t Defect Distance to Surface [A]\n")
for i, d in enumerate(dist_def):
    f.write('{} \t '.format(i * 10))
    for j in range(len(d)):
        if j != len(d) - 1:
            f.write('{} \t'.format(d[j]))
        else:
            f.write('{}\n'.format(d[j]))
f.close()


# Write surface to xyz file for VMD
print("Writing Isosurface to XYZ File")
xyz = open(iso_path + 'surface.xyz', 'w+')
for i in frames_arr:
    xyz.write('{}\n\n'.format(len(surface[0])))
    for j in range(len(surface[0])):
        xyz.write('H\t{:.3f}\t{:.6f}\t{:.6f}\n'.format(surface[i][j, 0], surface[i][j, 1], surface[i][j, 2]))
xyz.close()


# Write indices to file for VMD
def write_index(data, fname, type):
    f = open(iso_path + '{}.index'.format(fname), 'w+')
    if type == 'true_pos':
        for i in frames_arr:
            true_ice_frame = np.concatenate(([i * 10], mobile_ice_liq_idx[i][data[i]])).astype('int')
            for j in range(len(true_ice_frame)):
                f.write("{}".format(true_ice_frame[j]))
                if j != len(true_ice_frame) - 1:
                    f.write(" ")
            f.write("\n")
        f.close()
    elif type == 'false_pos':
        for i in frames_arr:
            n_false = mobile_ice_liq_idx[i][data[i]]
            n_false_hex = np.intersect1d(n_false, idx_hex[i])
            n_false_cubic = np.intersect1d(n_false, idx_cubic[i])
            n_false_int = np.intersect1d(n_false, idx_int[i])
            n_false_surf = np.intersect1d(n_false, idx_surf[i])
            n_false_ice = np.r_[n_false_hex, n_false_cubic, n_false_int, n_false_surf]
            false_ice_frame = np.concatenate(([i * 10], np.sort(n_false_ice))).astype('int')
            for j in range(len(false_ice_frame)):
                f.write("{}".format(false_ice_frame[j]))
                if j != len(false_ice_frame) - 1:
                    f.write(" ")
            f.write("\n")
        f.close()
    else:
        for i in frames_arr:
            n_true = mobile_ice_liq_idx[i][data[i]]
            n_true_liq = np.intersect1d(n_true, idx_liq[i])
            true_ice_frame = np.concatenate(([i * 10], np.sort(n_true_liq))).astype('int')
            for j in range(len(true_ice_frame)):
                f.write("{}".format(true_ice_frame[j]))
                if j != len(true_ice_frame) - 1:
                    f.write(" ")
            f.write("\n")
        f.close()


print("Writing Index Files")
write_index(true_idx_nn, 'NearestNeighbor', type='true_pos')
write_index(true_idx_mt, 'MollerTrumbore', type='true_pos')
write_index(false_idx_nn, 'NearestNeighborFalsePos', type='false_pos')
write_index(false_idx_mt, 'MollerTrumboreFalsePos', type='false_pos')
write_index(true_idx_nn, 'NearestNeighborFalseNeg', type='false_neg')
write_index(true_idx_mt, 'MollerTrumboreFalseNeg', type='false_neg')


# Write PDB file with Beta factor for coloring in VMD
# Color by distance from average interface height
print("Writing Isosurface to PDB File")
pdbtrj = iso_path + "isosurface.pdb"
u_surf = mda.Universe(iso_path + 'surface.xyz')
u_surf.add_TopologyAttr('tempfactors')

with mda.Writer(pdbtrj, multiframe=True, bonds=None, n_atoms=u_surf.atoms.n_atoms) as PDB:
    for ts in u_surf.trajectory:
        surf_trj = u_surf.select_atoms('type H').positions[:, 2]
        dist = surf_trj - np.average(surf_trj)
        u_surf.atoms.tempfactors = dist
        PDB.write(u_surf.atoms)


# Load CHILL+ Samples
chillplus_data = np.loadtxt('./chillplus/{}K/{:.1f}/{}K_{:.1f}_REP{}/chillplus_samples_{}K_{:.1f}_REP{}.out'.format(temp, phi, temp, phi, rep, temp, phi, rep))
x_sec_area = 11.734258148346557 * 11.030358970773076
t = chillplus_data[:, 0]


# Plot Lambda
def plot_lambda(data, type, scale='linear'):
    plt.clf()
    if type == 'MT':
        filler = r'{\text{{MT}}}'
    else:
        filler = r'{\text{{NN}}}'

    n_true_hex = np.zeros(n_frames)
    n_true_cubic = np.zeros(n_frames)
    n_true_int = np.zeros(n_frames)
    n_true_surf = np.zeros(n_frames)
    n_true_liq = np.zeros(n_frames)
    for i in frames_arr:
        n_true = mobile_ice_liq_idx[i][data[i]]
        n_true_hex[i] = np.intersect1d(n_true, idx_hex[i]).shape[0]
        n_true_cubic[i] = np.intersect1d(n_true, idx_cubic[i]).shape[0]
        n_true_int[i] = np.intersect1d(n_true, idx_int[i]).shape[0]
        n_true_surf[i] = np.intersect1d(n_true, idx_surf[i]).shape[0]
        n_true_liq[i] = np.intersect1d(n_true, idx_liq[i]).shape[0]
    plt.plot(t, lambda_chillplus, label=r'$\lambda_{\text{{CHILL+}}}$', color='k')
    plt.plot(t, n_true_mt, label=r'$\lambda_{}$'.format(filler), color=color_arr[0])
    plt.plot(t, n_true_hex, label='Hexagonal', color=color_arr[1])
    plt.plot(t, n_true_cubic, label='Cubic', color=color_arr[2])
    plt.plot(t, n_true_int, label='Interfacial', color=color_arr[3])
    plt.plot(t, n_true_surf, label='Surface', color=color_arr[4])
    plt.plot(t, n_true_liq, label='Liquid', color=color_arr[5])
    plt.xlabel("$t$ [ps]")
    plt.ylabel(r"$\lambda$")
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    if scale == 'log':
        plt.yscale('log')
        plt.savefig(iso_path + 'lambda_{}_{}.pdf'.format(type, scale), dpi=300, bbox_inches='tight')
    else:
        plt.savefig(iso_path + 'lambda_{}.pdf'.format(type), dpi=300, bbox_inches='tight')


plot_lambda(true_idx_mt, 'MT')
plot_lambda(true_idx_mt, 'MT', scale='log')
plot_lambda(true_idx_nn, 'NN')
plot_lambda(true_idx_nn, 'NN', scale='log')


# Plot Difference in Lambda
plt.clf()
plt.plot(t, lambda_chillplus - n_true_nn, label=r'$\lambda_{\text{{CHILL+}}} - \lambda_{\text{{NN}}}$', color=color_arr[0])
plt.plot(t, lambda_chillplus - n_true_mt, label=r'$\lambda_{\text{{CHILL+}}} - \lambda_{\text{{MT}}}$', color=color_arr[1])
plt.xlabel("$t$ [ps]")
plt.ylabel(r"$\Delta\lambda$")
plt.legend()
plt.savefig(iso_path + 'delta_lambda_chillplus.pdf', dpi=300)


# Plot Difference in True Lambda
plt.clf()
plt.plot(t, n_true_nn - n_true_mt, label=r'$\lambda_{\text{{NN}}} - \lambda_{\text{{MT}}}$', color=color_arr[0])
plt.axhline(np.average(n_true_nn - n_true_mt), label=r'$\langle\lambda_{{\text{{NN}}}} - \lambda_{{\text{{MT}}}}\rangle = {:.2f} \pm {:.2f}$'.format(np.average(n_true_nn - n_true_mt), np.std(n_true_nn - n_true_mt)), color='k')
plt.xlabel("$t$ [ps]")
plt.ylabel(r"$\Delta\lambda$")
plt.legend()
plt.savefig(iso_path + 'delta_lambda_true.pdf', dpi=300)


# Plot Area
plt.clf()
plt.plot(t, np.array(area) / 100, color=color_arr[0], label='Isosurface Area')
plt.axhline(x_sec_area, 0, 5000, color='k', label='Cross-Sectional Area of Box')
plt.xlabel("$t$ [ps]")
plt.ylabel(r"Area [$\text{nm}^2$]")
plt.legend()
plt.savefig(iso_path + 'area.pdf', dpi=300)


# Plot Volume
plt.clf()
plt.plot(t, np.array(vol) / 1000, color=color_arr[0])
plt.xlabel("$t$ [ps]")
plt.ylabel(r"Volume [$\text{nm}^3$]")
plt.savefig(iso_path + 'volume.pdf', dpi=300)


# Plot Defect Data
plt.clf()
avg_dist = np.hstack(dist_def)
for i, d in enumerate(dist_def):
    if ~np.isnan(d)[0]:
        t = np.repeat(i * 10, repeats=len(d))
        plt.scatter(t, d, color=color_arr[0])
plt.axhline(0, color='k', linestyle=':')
plt.axhline(np.average(avg_dist[~np.isnan(avg_dist)]), label=r'$\langle r_{\text{Surf}} - r_{\text{Def}}\rangle$', color=color_arr[1], linestyle='--')
plt.axhline(np.average(avg_dist[~np.isnan(avg_dist)]) + np.std(avg_dist[~np.isnan(avg_dist)]), color=color_arr[1])
plt.axhline(np.average(avg_dist[~np.isnan(avg_dist)]) - np.std(avg_dist[~np.isnan(avg_dist)]), color=color_arr[1])
plt.xlabel("$t$ [ps]")
plt.ylabel(r"Defect Distance to Surface [\AA]")
plt.legend()
plt.savefig(iso_path + 'defect_distance.pdf', dpi=300)


plt.clf()
fig = plt.figure()
ax = fig.add_subplot(111)
vol_def_sum = np.zeros(frames_arr.shape, dtype=np.float32)
n_defects = np.zeros(frames_arr.shape, dtype=int)
for i, v in enumerate(vol_def):
    vol_def_sum[i] = np.sum(v)
    if v[0] != 0:
        n_defects[i] = len(v)

ax.plot(10 * frames_arr, vol_def_sum, color=color_arr[0])
plt1 = ax.axhline(np.average(vol_def_sum[np.nonzero(vol_def_sum)]), label=r'$\langle V_{{\text{{Def, Tot}}}}\rangle = {:.3f}$ \AA$^3$'.format(np.average(vol_def_sum[np.nonzero(vol_def_sum)])), color='k', linestyle='-.')
ax2 = ax.twinx()
ax2.plot(10 * frames_arr, n_defects, color=color_arr[1], linestyle=':')
plt2 = ax2.axhline(np.average(n_defects), label=r'$\langle N_{{\text{{Def}}}}\rangle = {:.3f}$'.format(np.average(n_defects)), color='k', linestyle='--')
ax.set_xlabel("$t$ [ps]")
ax.set_ylabel(r"$V_{\text{{Def, Tot}}}$ [\AA$^3$]", color=color_arr[0])
ax2.set_ylabel(r"$N_{\text{{Def}}}$", color=color_arr[1])
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines + lines2, labels + labels2, bbox_to_anchor=(0.5, -0.25), loc=8, ncol=2)
plt.savefig(iso_path + 'defect_volume.pdf', dpi=300, bbox_inches='tight')


plt.clf()
vol_def_sum_norm = np.zeros(frames_arr.shape, dtype=np.float32)
n_defects = np.zeros(frames_arr.shape, dtype=int)
for i, v in enumerate(vol_def):
    vol_def_sum_norm[i] = np.sum(v) / vol[i]

plt.plot(10 * frames_arr, vol_def_sum_norm, color=color_arr[0])
plt.axhline(np.average(vol_def_sum_norm[np.nonzero(vol_def_sum_norm)]), label=r'$\left\langle \frac{{V_{{\text{{Def, Tot}}}}}}{{V_{{\text{{Bulk Ice}}}}}}\right\rangle = {:.3e}$'.format(np.average(vol_def_sum_norm[np.nonzero(vol_def_sum_norm)])), color='k', linestyle=':')
plt.ylabel(r"$\frac{V_{\text{{Def, Tot}}}}{V_{\text{{Bulk Ice}}}}$")
plt.legend()
plt.savefig(iso_path + 'defect_volume_normalized.pdf', dpi=300, bbox_inches='tight')


# Plot interface location
avg_surf = []
std_surf = []
for i, v in enumerate(verts_new):
    avg_surf.append(np.average(v[0][:, 2] / 10))
    std_surf.append(np.std(v[0][:, 2] / 10))
avg_surf = np.array(avg_surf)
std_surf = np.array(std_surf)
plt.clf()
plt.plot(t, avg_surf, color=color_arr[0], label='Average Interface Height')
plt.fill_between(t, avg_surf + std_surf, avg_surf - std_surf, color=color_arr[0], alpha=0.5)
plt.xlabel("$t$ [ps]")
plt.ylabel(r"Interface Height [nm]")
plt.legend()
plt.savefig(iso_path + 'height.pdf', dpi=300)


# Plot false positives
def plot_false_pos(data, type):
    plt.clf()
    if type == 'MT':
        filler = r'{\text{{MT}}}'
    else:
        filler = r'{\text{{NN}}}'

    n_false_tot = np.zeros(n_frames)
    n_false_hex = np.zeros(n_frames)
    n_false_cubic = np.zeros(n_frames)
    n_false_int = np.zeros(n_frames)
    n_false_surf = np.zeros(n_frames)
    for i in frames_arr:
        n_false = mobile_ice_liq_idx[i][data[i]]
        n_false_tot[i] = n_false.shape[0] - np.intersect1d(n_false, idx_liq[i]).shape[0]
        n_false_hex[i] = np.intersect1d(n_false, idx_hex[i]).shape[0]
        n_false_cubic[i] = np.intersect1d(n_false, idx_cubic[i]).shape[0]
        n_false_int[i] = np.intersect1d(n_false, idx_int[i]).shape[0]
        n_false_surf[i] = np.intersect1d(n_false, idx_surf[i]).shape[0]
    plt.plot(t, n_false_tot, label='Total', color='k')
    plt.plot(t, n_false_hex, label='Hexagonal', color=color_arr[0])
    plt.plot(t, n_false_cubic, label='Cubic', color=color_arr[1])
    plt.plot(t, n_false_int, label='Interfacial', color=color_arr[2])
    plt.plot(t, n_false_surf, label='Surface', color=color_arr[3])
    plt.xlabel("$t$ [ps]")
    plt.ylabel(r"Number of False Positives From $\lambda_{}$".format(filler))
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.savefig(iso_path + 'false_positives_{}.pdf'.format(type), dpi=300, bbox_inches='tight')


plot_false_pos(false_idx_mt, 'MT')
plot_false_pos(false_idx_nn, 'NN')


def plot_near_surface(idx_data, dist_data, type, cutoff=None, maw=None):
    plt.clf()
    if type == 'MT':
        filler = r'{\text{{MT, False+}}}'
    else:
        filler = r'{\text{{NN, False+}}}'

    dist_hex = np.zeros((n_frames, 2))
    dist_int = np.zeros((n_frames, 2))
    dist_liq = np.zeros((n_frames, 2))
    for i in frames_arr:
        if cutoff is None:
            false_filtered_idx = idx_data[i]
            false_filtered_dist = dist_data[i]
        else:
            cond = dist_data[i] <= cutoff
            false_filtered_idx = idx_data[i][cond].astype(int)
            false_filtered_dist = dist_data[i][cond]

        dist_hex[i, 0] = np.average(false_filtered_dist[np.in1d(mobile_ice_liq_idx[i][false_filtered_idx], idx_hex[i])])
        dist_int[i, 0] = np.average(false_filtered_dist[np.in1d(mobile_ice_liq_idx[i][false_filtered_idx], idx_int[i])])
        dist_liq[i, 0] = np.average(false_filtered_dist[np.in1d(mobile_ice_liq_idx[i][false_filtered_idx], idx_liq[i])])
        dist_hex[i, 1] = np.std(false_filtered_dist[np.in1d(mobile_ice_liq_idx[i][false_filtered_idx], idx_hex[i])])
        dist_int[i, 1] = np.std(false_filtered_dist[np.in1d(mobile_ice_liq_idx[i][false_filtered_idx], idx_int[i])])
        dist_liq[i, 1] = np.std(false_filtered_dist[np.in1d(mobile_ice_liq_idx[i][false_filtered_idx], idx_liq[i])])
    hex_bounds = np.c_[dist_hex[:, 0] + dist_hex[:, 1], dist_hex[:, 0] - dist_hex[:, 1]]
    hex_bounds[hex_bounds < 0] = 0
    int_bounds = np.c_[dist_int[:, 0] + dist_int[:, 1], dist_int[:, 0] - dist_int[:, 1]]
    int_bounds[int_bounds < 0] = 0
    liq_bounds = np.c_[dist_liq[:, 0] + dist_liq[:, 1], dist_liq[:, 0] - dist_liq[:, 1]]
    liq_bounds[liq_bounds < 0] = 0
    plt.plot(t, dist_hex[:, 0], label='Hexagonal', color=color_arr[0])
    plt.fill_between(t, hex_bounds[:, 0], hex_bounds[:, 1], color=color_arr[0], alpha=0.5)
    plt.plot(t, dist_int[:, 0], label='Interfacial', color=color_arr[1])
    plt.fill_between(t, int_bounds[:, 0], int_bounds[:, 1], color=color_arr[1], alpha=0.5)
    plt.plot(t, dist_liq[:, 0], label='Liquid', color=color_arr[2])
    plt.fill_between(t, liq_bounds[:, 0], liq_bounds[:, 1], color=color_arr[2], alpha=0.5)
    if maw is not None:
        hex_ma = np.convolve(dist_hex[:, 0], np.ones(maw), 'valid') / maw
        int_ma = np.convolve(dist_int[:, 0], np.ones(maw), 'valid') / maw
        liq_ma = np.convolve(dist_liq[:, 0], np.ones(maw), 'valid') / maw
        plt.plot(t[maw - 1:], hex_ma, color='k', label='Hexagonal MA', linestyle=':')
        plt.plot(t[maw - 1:], int_ma, color='k', label='Interfacial MA', linestyle='--')
        plt.plot(t[maw - 1:], liq_ma, color='k', label='Liquid MA', linestyle='-.')
    plt.xlabel("$t$ [ps]")
    plt.ylabel(r"$\lambda_{} \ \langle dr_{{\text{{Surf}}}} \rangle$ [\AA]".format(filler))
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.savefig(iso_path + 'near_surface_{}_{}.pdf'.format(type, cutoff), dpi=300, bbox_inches='tight')


plot_near_surface(false_idx_mt, false_dist_mt, 'MT', maw=20)
plot_near_surface(false_idx_nn, false_dist_nn, 'NN', maw=20)
plot_near_surface(false_idx_mt, false_dist_mt, 'MT', cutoff=3.5, maw=20)
plot_near_surface(false_idx_nn, false_dist_nn, 'NN', cutoff=3.5, maw=20)


# Plot false negatives
def plot_false_neg(data, type, maw=None):
    plt.clf()
    if type == 'MT':
        filler = r'{\text{{MT}}}'
    else:
        filler = r'{\text{{NN}}}'

    n_false_neg = np.zeros(n_frames)
    for i in frames_arr:
        n_true = mobile_ice_liq_idx[i][data[i]]
        if i == 0:
            false_neg_idx = np.intersect1d(n_true, idx_liq[i])
        else:
            false_neg_idx = np.append(false_neg_idx, np.intersect1d(n_true, idx_liq[i]))
        n_false_neg[i] = np.intersect1d(n_true, idx_liq[i]).shape[0]

    plt.plot(t, n_false_neg, color=color_arr[0])
    if maw is not None:
        ma = np.convolve(n_false_neg, np.ones(maw), 'valid') / maw
        plt.plot(t[maw - 1:], ma, color='k', label='Moving Average', linestyle=':')
        plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.xlabel("$t$ [ps]")
    plt.ylabel(r"Number of False Negatives From $\lambda_{}$".format(filler))
    plt.savefig(iso_path + 'false_negatives_{}.pdf'.format(type), dpi=300, bbox_inches='tight')


    plt.clf()
    value, counts = np.unique(false_neg_idx, return_counts=True)
    plt.hist(counts, bins=np.linspace(1, n_frames, num=n_frames), color=color_arr[0])
    plt.xlabel("Number of False Negative Appearances")
    plt.ylabel("Counts")
    plt.yscale('log')
    plt.savefig(iso_path + 'false_negatives_distribution_{}.pdf'.format(type), dpi=300, bbox_inches='tight')


plot_false_neg(true_idx_mt, 'MT', maw=20)
plot_false_neg(true_idx_nn, 'NN', maw=20)
