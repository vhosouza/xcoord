%%
figure;
hold on
qn = coils_rot.QN(1, 1:3)/norm(coils_rot.QN(1, 1:3));

cface = transf(1:3, 1)';
cdir = transf(1:3, 2)';
cn = transf(1:3, 3)';

quiver3(0, 0, 0, cface(1), cface(2), cface(3), 1, 'Color', 'r');
quiver3(0, 0, 0, cdir(1), cdir(2), cdir(3), 1, 'Color', 'g');
quiver3(0, 0, 0, cn(1), cn(2), cn(3), 1, 'Color', 'b');

quiver3(0, 0, 0, qn(1), qn(2), qn(3), 1, 'Color', 'm');
hold off

%%

ang1 = rad2deg(acos(dot(cface, cdir)));
ang2 = rad2deg(acos(dot(cface, cn)));
ang3 = rad2deg(acos(dot(cdir, cn)));
fprintf("\nThe angle between face and dir is: %.2f\n", ang1);
fprintf("The angle between face and normal is: %.2f\n", ang2);
fprintf("The angle between dir and normal is: %.2f\n", ang3);