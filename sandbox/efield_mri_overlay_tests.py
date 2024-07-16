import matplotlib.pyplot as plt
import numpy as np

# %%
indices = np.lexsort((efield_rtp_extended[:, 3], efield_rtp_extended[:, 2], efield_rtp_extended[:, 1], efield_rtp_extended[:, 0]))
efield_rtp_extended = efield_rtp_extended[indices]

# %%
coordinate = ['r', 't', 'p', 'ef']
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(6, 6))
for n, ax in enumerate(axs.flatten()):
    #ax.scatter(np.arange(0, efield_rtp_extended.shape[0]), efield_rtp_extended[:, n], s=3)
    ax.plot(np.arange(0, efield_rtp_extended.shape[0]), efield_rtp_extended[:, n], lw=0.5)
    ax.set_title(coordinate[n])
    ax.set_xlim(0, 200)
fig.tight_layout()
plt.show()

# %%
plot_var = xyz_rtp_masked
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
ax.scatter(np.arange(0, plot_var.shape[0]), plot_var[:, 0], s=3)
#ax.set_ylim(60, 80)
fig.tight_layout()
plt.show()

# %%
x = xyz_rtp_masked[::100, 1]
y = xyz_rtp_masked[::100, 2]
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
#ax.scatter(x, y, s=3)
ax.plot(x, y, lw=0.5)
#ax.set_ylim(60, 80)
fig.tight_layout()
plt.show()

# %%
plot_var = efield_rtp_interp
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(3, 3))
ax.imshow(plot_var, origin='lower')
fig.tight_layout()
plt.show()

# %%
plot_var = xyz_rtp_masked[efield_rtp_interp == 0.]
plot_value = efield_rtp_interp[efield_rtp_interp == 0.]/100.
plot_scatter = efield_rtp
plot_scatter_value = efield_rtp[:, 3]/100.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax.plot_trisurf(plot_var[:, 1], plot_var[:, 2], plot_value, antialiased=True)
ax.scatter3D(plot_var[:, 1], plot_var[:, 2], plot_value, color='k', marker='x')
ax.scatter3D(plot_scatter[:, 1], plot_scatter[:, 2], plot_scatter_value, color='r', marker='.')
#ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.set_xlim([0, 3])
# ax.set_ylim([1, 4])
# ax.set_zlim([0, 1])
# Set view angle
elevation_angle = 90  # Up/down
azimuth_angle = 0  # Left/right
ax.view_init(elev=elevation_angle, azim=azimuth_angle)
fig.tight_layout()
plt.show()

# %%
efield_points = 1e3*imf.polar_to_cartesian(efield_rtp[:, :3])
xp, yp, zp = efield_points[:, 0], efield_points[:, 1], efield_points[:, 2]
up = efield_rtp[:, 3]
#u = efield_rtp[:, 3]
#efield_xyz = imf.polar_to_cartesian(xyz_rtp_masked[:, :3])
#efield_xyz_plot = image_xyz_interp[:, :3]
efield_xyz_plot = image_ijk_interp[:, :3]
u = efield_interp

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = efield_xyz_plot[:, 0], efield_xyz_plot[:, 1], efield_xyz_plot[:, 2]
ef_max = np.squeeze(efield_interp > 0.97*efield_interp.max())
xmax, ymax, zmax = efield_xyz_plot[ef_max, 0], efield_xyz_plot[ef_max, 1], efield_xyz_plot[ef_max, 2]
# norm_p = colors.Normalize(up.min(), up.max())
# colormap_p = cm.copper
# mappable_p = cm.ScalarMappable(norm=norm_p, cmap=colormap_p)
# mappable_p.set_array(up)

norm = colors.Normalize(u.min(), u.max())
colormap = cm.cividis
mappable = cm.ScalarMappable(norm=norm, cmap=colormap)
mappable.set_array(u)
#surf = ax.plot_trisurf(x, y, z, facecolors=mappable.to_rgba(u), cmap=colormap, linewidth=1, shade=False)
#sc = ax.scatter3D(xp, yp, zp, facecolors=mappable_p.to_rgba(up), cmap=colormap_p)
sc = ax.scatter3D(x, y/2, z/2, facecolors=mappable.to_rgba(u), alpha=0.1)
sc = ax.scatter3D(49, 314/2, 388/2, color='m', marker='o', s=15)
sc = ax.scatter3D(xmax, ymax/2, zmax/2, color='k', marker='o', s=15)
sc = ax.plot(np.linspace(0, image_shape[0], 100),
             np.zeros(100) + 314/2,
             np.zeros(100) + 388/2, '-r')
sc = ax.plot(np.zeros(100) + 49,
             np.linspace(0, image_shape[1]/2, 100),
             np.zeros(100) + 388/2, '-b')
sc = ax.plot(np.zeros(100) + 49,
             np.zeros(100) + 314/2,
             np.linspace(0, image_shape[2]/2, 100), '-g')
elevation_angle = 30  # Up/down
azimuth_angle = 90  # Left/right # 120 good to see the dlpfc
ax.view_init(elev=elevation_angle, azim=azimuth_angle)
ax.set_aspect('equal')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_xlim(0, image_shape[0])
ax.set_ylim(0, image_shape[1]/2)
ax.set_zlim(0, image_shape[2]/2)
fig.tight_layout()
plt.show()
