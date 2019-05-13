
function coils_rot = rotate_coil(coils, vis)

% To match transformations from Nexstim navigation to world space of mesh
% coordinates generated with SimNIBS, the coil should have the following
% initial allignment.
% x axis: longitudinal coil axis pointing forward
% y axis: the coil normal pointing downwards
% z axis: pointing to the left wing of the coil
% origin: center of the windings
% Arguments
% coils: structure with QN as coil normal coordinates and QP as coil points
% coordinates
% vis: visualization flag for plotting (true or false)
% 
% (c) Victor Souza (2019) victor.souza@aalto.fi
% Date: 26.4.2019

coils_rot = coils;

rotz = makehgtform('zrotate', pi/2);
rotz = rotz(1:3, 1:3);

rotx = makehgtform('xrotate', pi/2);
rotx = rotx(1:3, 1:3);

coils_rot.QP(1:end, :) = (rotx*rotz*coils.QP(1:end, :)')';
coils_rot.QN(1:end, :) = (rotx*rotz*coils.QN(1:end, :)')';
coils_rot.QN = -1*coils_rot.QN;
coils_rot.transform = 'Coil points and normals rotated first pi/2 around Z, then pi/2 around X, and then normals inverted';

if vis
    n = 10;  % plot every 10th point to avoid too much information
    figure;
    hold on
    plot3(coils_rot.QP(:, 1), coils_rot.QP(:, 2), coils_rot.QP(:, 3), '.');
%     quiver3(coils_rot.QP(:, 1), coils_rot.QP(:, 2), coils_rot.QP(:, 3),...
%         coils_rot.QN(:, 1), coils_rot.QN(:, 2), coils_rot.QN(:, 3), 1e-1);
    quiver3(coils_rot.QP(1:n:end, 1), coils_rot.QP(1:n:end, 2), coils_rot.QP(1:n:end, 3),...
        coils_rot.QN(1:n:end, 1), coils_rot.QN(1:n:end, 2), coils_rot.QN(1:n:end, 3), 1e-1);
    hold off
    xlabel('x');
    ylabel('y');
    zlabel('z');
end

end