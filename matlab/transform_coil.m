
function coils_new = transform_coil(coils, m_affine, transl, coords, vis)

% Transform coil to position and orientation given by a navigation system
% 
% To match transformations from Nexstim navigation to world space of mesh
% coordinates generated with SimNIBS, the coil should have the following
% initial allignment.
% x axis: vector pointing to the right wing of the coil (coil face)
% y axis: longitudinal coil axis pointing forward (coil direction)
% z axis: the coil normal pointing upwards, away from the head (coil normal)
% origin: center of the windings
% 
% Input:
% coils: structure with QN (m x 3) as coil normal coordinates and
% QP (m x 3) as coil points with m as the number of points and 3 as for x,
% y and z coordinates
% m_affine: 4x4 affine transformation matrix to be applyed as m_affine*p in
% which p is the point coordinates arranged in a 4x1 array
% transl: optional translation before applying the affine transformations.
% This can be used when coil model is shifted in some direction, e.g. z,
% and user whish to allign the coil windings to the origin of coordinate
% system. Example: [0, 0, min(coils.QP(:, end))] if one wants to translate
% to the minimum z coordinate of all points.
% coords: set of additional points to be visualized, such as fiducials and
% coil location
% vis: visualization flag for plotting (true or false)
% 
% Output:
% coils_new: structure with QN (m x 3) as rotated coil normal and
% QP (m x 3) as affine-transformed coil points with m as the number of
% points and 3 as for x, y and z coordinates. The output structure keeps
% all other fields of the input coils structure and add the field
% "transform" to identify that a transformation was applied.
% 
% (c) Victor Souza (2019) victor.souza@aalto.fi
% Date: 26.4.2019
% Latest: 5.5.2019

% translate coil to a different initial position before applying navigation
% transformations
coils.QP(1:end, :) = coils.QP(1:end, :) - transl;
% add 1 to each coordiante point to enable affine 4x4 matrix multiplication
coils.QP = [1000*coils.QP ones(size(coils.QP, 1), 1)];
coils_new = coils;

% rotate and translate coil according to affine transformation from
% navigation system
coils_new.QP(1:end, :) = (m_affine*coils.QP(1:end, :)')';
coils_new.QP = coils_new.QP(:, 1:3);

% apply rotation to coil normals
rot = m_affine(1:3, 1:3);
coils_new.QN(1:end, :) = (rot*coils.QN(1:end, :)')';

coils_new.transform = 'Coil points and normals aligned to coordinates exported by navigation system';

if vis
    
    % scale the vectors when plotting
    scale = 10;
    % unitary vectors of affine transformation
    cface = m_affine(1:3, 1)';
    cdir = m_affine(1:3, 2)';
    cn = m_affine(1:3, 3)';
    orig = m_affine(1:3, 4)';

    
    % create a refrence plane for better visualization
    [x, y] = meshgrid(-10:2:10);
    z = zeros(size(x));
    w = ones(size(x));
    plane = [x(:), y(:), z(:), w(:)];
    plane_rot = (m_affine*plane(1:end, :)')';
    
    % unitary coil normal vector after rotation
    qn = (coils_new.QN(1, 1:3)/norm(coils_new.QN(1, 1:3)));
    
    % display the angle between vectors to confirmat that coil normal is
    % rotated accordingly
    ang1 = rad2deg(acos(dot(cface, cdir)));
    ang2 = rad2deg(acos(dot(cface, cn)));
    ang3 = rad2deg(acos(dot(cdir, cn)));
    ang4 = rad2deg(acos(dot(qn, cn)));
    
    fprintf("\nThe angle between face and dir is: %.2f\n", ang1);
    fprintf("The angle between face and normal is: %.2f\n", ang2);
    fprintf("The angle between dir and cn is: %.2f\n", ang3);
    fprintf("The angle between cn and qn is: %.2f\n", ang4);
    
    n = 1;  % plot every 10th point to avoid too much information

    figure;
    hold on
    % additional coordiantes
    plot3(coords(1:end, 1), coords(1:end, 2), coords(1:end, 3), 'r.');
    % coil points
    plot3(coils_new.QP(1:n:end, 1), coils_new.QP(1:n:end, 2), coils_new.QP(1:n:end, 3), '.');
    % reference place
    plot3(plane_rot(:, 1), plane_rot(:, 2), plane_rot(:, 3), 'k.');
    
    % coil model normals
    quiver3(coils_new.QP(1:n:end, 1), coils_new.QP(1:n:end, 2), coils_new.QP(1:n:end, 3),...
        coils_new.QN(1:n:end, 1), coils_new.QN(1:n:end, 2), coils_new.QN(1:n:end, 3), 0.5, 'Color', 'm');
%     quiver3(orig(1), orig(2), orig(3), qn(1), qn(2), qn(3), scale*1.5, 'Color', 'm');

    % vectors of affine coordinate system
    quiver3(orig(1), orig(2), orig(3), cface(1), cface(2), cface(3), scale, 'Color', 'r');
    quiver3(orig(1), orig(2), orig(3), cdir(1), cdir(2), cdir(3), scale, 'Color', 'g');
    quiver3(orig(1), orig(2), orig(3), cn(1), cn(2), cn(3), scale, 'Color', 'b');
    
    % vectors of world coordinate system
    quiver3(0, 0, 0, 1, 0, 0, 10, 'Color', 'r');
    quiver3(0, 0, 0, 0, 1, 0, 10, 'Color', 'g');
    quiver3(0, 0, 0, 0, 0, 1, 10, 'Color', 'b');

    % sphere to represent the cortex with 70 mm radius
%     [x, y, z] = sphere(20);
%     surf(70*x, 70*y, 70*z);
    
    hold off
    xlabel('x');
    ylabel('y');
    zlabel('z');
end

end