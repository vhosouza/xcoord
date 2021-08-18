%%

SHOW_SCALP = true;
LOAD_IMG = false;
SHOW_PTS = true;


data_path = 'P:\tms_eeg\mTMS\projects\2019 EEG-based target automatization\Analysis\EEG electrode transformation\Locations of interest in Nexstim coords';
fname_coord_nexstim = 'EEGTA04_POI_coords_Nexstim.mat'; 
fname_coord_mri = 'EEGTA04_POI_coords_Nexstim_MRI.csv';
scalp_dir = 'P:\tms_eeg\mTMS\projects\2019 EEG-based target automatization\Analysis\EEG electrode transformation';
fname_scalp = 'EEGTA04_scalp_mesh.mat';

% data_path = 'P:\tms_eeg\mTMS\projects\2019 EEG-based target automatization\Analysis\EEG electrode transformation';
% scalp_dir =data_path;
% fname_coord_nexstim = 'EEGTA04_electrode_locations_Nexstim.mat'; 
% fname_coord_mri = 'EEGTA04_electrode_locations_MRI.csv';
% fname_img = 'EEGTA04.nii';

eeg_next_filename = fullfile(data_path, fname_coord_nexstim);
eeg_mri_filename = fullfile(data_path, fname_coord_mri);

eeg_nexstim = load(eeg_next_filename);
% eeg_nexstim = eeg_nexstim.digitized_points;
eeg_nexstim = eeg_nexstim.POI_coords_Nexstim;

eeg_mri = readmatrix(eeg_mri_filename);

if LOAD_IMG
    img_filename = fullfile(data_path, fname_img);
    img_data = niftiinfo(img_filename);
    matrix = img_data.Transform.T;

    % idx = [3 1 2 4]; % match affine order of python canonical
    % this almost match the correct location, there's just some offset from
    % origin. might require only some translatation relative to image
    % dimensions, similar to what is done in python
    idx = [3 2 1 4];
    matrix_per = matrix(idx,:);
    matrix_per(:, 2) = -1*matrix_per(:, 2);
    matrix_per(:, 1) = -1*matrix_per(:, 1);
    % ---
    
    eeg_nexstim_w = [eeg_nexstim, ones(size(eeg_nexstim, 1), 1)];
    eeg_next_mri = eeg_nexstim_w*matrix_per;
end

if SHOW_SCALP
    scalp_filename = fullfile(scalp_dir, fname_scalp);
    scalp = load(scalp_filename);
    scalp = scalp.scalp;
end

%%
deffacecolor=[1 .7 .7];
deffacealpha=.3;
defview=[-90 0];

figure(1);
ah = gca;
hold on

if SHOW_SCALP
    hp = patch(ah, 'faces', scalp.e,'vertices', 1e3*scalp.p,'facecolor',deffacecolor,'edgecolor','none',...,
        'facealpha',deffacealpha);
end

if SHOW_PTS
    plot3(ah, eeg_nexstim(:, 1), eeg_nexstim(:, 2), eeg_nexstim(:, 3), 'og');
    plot3(eeg_mri(:, 1), eeg_mri(:, 2), eeg_mri(:, 3), 'ok')
    % plot directly from the loaded MRI space coordinates
%     plot3(coords_mri(:, 1), coords_mri(:, 2), coords_mri(:, 3), 'xr')
    % plot directly from the original Nexstim space coordiantes
%     plot3(POI_coords_Nexstim(:, 1), POI_coords_Nexstim(:, 2), POI_coords_Nexstim(:, 3), 'xb')
    % plot3(eeg_next_mri(:, 1), eeg_next_mri(:, 2), eeg_next_mri(:, 3), 'or')
end

hold off;
view(defview);
axis tight equal off; material dull; lighting gouraud;
%%


% function [scale, shear, angles, translate, perspective] = decompose_matrix(matrix)
%     Return sequence of transformations from transformation matrix.
% 
%     matrix : array_like
%         Non-degenerative homogeneous transformation matrix
% 
%     Return tuple of:
%         scale : vector of 3 scaling factors
%         shear : list of shear factors for x-y, x-z, y-z axes
%         angles : list of Euler angles about static x, y, z axes
%         translate : translation vector along x, y, z axes
%         perspective : perspective partition of matrix
% 
%     Raise ValueError if matrix is of wrong type or degenerative.
% 
%     >>> T0 = translation_matrix([1, 2, 3])
%     >>> scale, shear, angles, trans, persp = decompose_matrix(T0)
%     >>> T1 = translation_matrix(trans)
%     >>> numpy.allclose(T0, T1)
%     True
%     >>> S = scale_matrix(0.123)
%     >>> scale, shear, angles, trans, persp = decompose_matrix(S)
%     >>> scale[0]
%     0.123
%     >>> R0 = euler_matrix(1, 2, 3)
%     >>> scale, shear, angles, trans, persp = decompose_matrix(R0)
%     >>> R1 = euler_matrix(*angles)
%     >>> numpy.allclose(R0, R1)
%     True
% 

M = matrix';
if abs(M(4, 4)) < eps
    error("M[4, 4] is zero");
end

M = M ./ M(4, 4);
P = M;
P(:, 4) = [0.0, 0.0, 0.0, 1.0];

if det(P) == 0
    error("matrix is singular");
end

scale = zeros(1, 3);
shear = zeros(1, 3);
angles = zeros(1, 3);

if any(abs(M(1:end-1, end)) > eps)
    perspective = dot(M(:, end), inv(P'));
    M(:, end) = [0.0, 0.0, 0.0, 1.0];
else
    perspective = [0.0, 0.0, 0.0, 1.0];
end

%%
translate = M(end, 1:end);
M(end, 1:end) = 0.0;

row = M(1:end, 1:end);
scale(1) = norm(row(1, :));
row(1, :) = row(1)/scale(1);
shear(1) = dot(row(1, :), row(2, :));

row(2, :) = row(2, :) - row(1, :) * shear(1);
scale(2) = norm(row(2, :));
row(2, :) = row(2, :) / scale(2);
shear(1) = shear(1)/scale(2);
shear(2) = dot(row(1, :), row(3, :));

row(3, :) = row(3, :) - row(1, :) * shear(2);
shear(3) = dot(row(2, :), row(3, :));
row(3, :) = row(3, :) - row(2, :) / shear(3);
scale(3) = norm(row(3, :));
row(3, :) = row(3, :) / scale(3);
shear(2:end) = shear(2:end)/scale(3);

 
if dot(row(1, :), cross(row(1, :), row(2, :))) < 0
    scale = -scale;
    row = -row;
end

angles(1) = asin(-row(1, 3));
if cos(angles(2))
    angles(1) = atan2(row(2, 3), row(3, 3));
    angles(3) = atan2(row(1, 2), row(1, 1));
else
    angles(1) = atan2(row(2, 1), row(2, 2));
    angles(1) = atan2(-row(3, 2), row(2, 2));
    angles(3) = 0.0;
end

    
% end