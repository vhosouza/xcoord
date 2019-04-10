# %%
#Loading the coil coordinates from NBS eXimia session file
import re
import os
from itertools import zip_longest

# %%
data_dir = os.environ['OneDriveConsumer']


filename = data_dir + '\\data\\nexstim_coord\\s01_eximia_coords.txt'
with open(filename, 'r') as f:
    x = f.readlines()

x = [e for e in x if e != '\n']
x = [e.replace('\n', '') for e in x]
x = [e[1:] if e[0] == '\t' else e for e in x]

# print(x)
for line in x:
    print(line.encode('unicode_escape'))
subj = x[1]

# %%
print(x[0])
subj = x[1]
print(subj)

nasion = x[10]
leftear = x[11]
rightear = x[12]

print(nasion)

# %%

n_data = nasion.split('\t')
n_data_rs = nasion.rsplit('\t')
print(n_data)
print(n_data_rs)

# %%


def find_pattern(text, pat):
    seq = []
    for t in text:
        # print('Looking for "%s" in "%s" ->' % (pat, t), end=' ')

        if re.search(pat, t):
            # print('found a match!')
            seq = t
            break
        else:
            pass
            # print('no match!')
    return seq


# %%
patterns = ['MRI landmark: Nose/Nasion', 'MRI landmark: Left ear', 'MRI landmark: Right ear']
coord_fid = [[], [], []]
id_fid = 0
for pattern in patterns:
    # print('%s' % pattern, end=' ')
    seq_found = find_pattern(x, pattern)
    if seq_found:
        for token in seq_found.split('\t'):
            try:
                coord_fid[id_fid].append(float(token))
            except ValueError:
                pass
    else:
        pass

    print('%s: %.1f %.1f %.1f' % (pattern, *coord_fid[id_fid]))
    id_fid += 1

# %%

pattern = 'Time'
# data = [[] for id_l in range(6)]
data = []
data_csv = []
id_fid = 0
# print('%s' % pattern, end=' ')
id_line = 0
for s in x:
    seq_found = re.search(pattern, s)
    if seq_found:
        for id_l in range(5):
            data_line = x[id_fid + id_l].replace('\n', '')
            data.append(data_line.split('\t'))
            print(data_line.split('\t'))
            data_csv.append(x[id_fid + id_l].replace('\t', 'NA'))
        break
    id_fid += 1

print(data)
# print(x[id_fid])
data_new = data.copy()

# %%
pattern = 'Time'
for n, s in enumerate(x):
    seq_found = re.search(pattern, s)
    if seq_found:
        final = [z.split('\t') for z in x[n:]]
        stim_info = [t for t in zip_longest(*final)]
        break

# %%
pattern = 'Landmarks'
for n, s in enumerate(x):
    seq_found = re.search(pattern, s)
    if seq_found:
        final = [z.split('\t') for z in x[n+1:n+8]]
        fids = [t for t in zip_longest(*final)]
        break

reorder = [3, 0, 1, 2]
fids = [fids[s] for s in reorder]

# %%
# labels = [fids[-1][0], fids[0][0], fids[1][0], fids[2][0]]
fids_mri = [[] for id_l in range(len(fids[0]))]
# fids_mri = []
for s in fids:
    for n, l in enumerate(s):
        fids_mri[n].append(l)

# %%
rel_data = fids_mri.copy()

for n, s in enumerate(stim_info):
    if s[0] == 'Coil' or (s[0] == 'EF max.' and s[1] == 'Loc.'):
        rel_data.append([s[0] + ' ' + s[1], stim_info[n][-1], stim_info[n+1][-1], stim_info[n+2][-1]])

# %%



# %%
# fids_mri = [s for n in s for s in fids]

# for n in range(4):
#     fids_mri[n] = [fids[-1][n], fids[0][n], fids[1][n], fids[2][n]]



# %%

a = []
for im in data_new:
    a.append(len(im))
max_len = max(a)

for n, im in enumerate(data):
    if len(im) < max_len:
        # data_extra = ['' for id_l in range(max_len - len(im))]
        data_new[n].extend(['NA' for id_l in range(max_len - len(im))])

for n, i in enumerate(a):
    if i == 1:
        a[n] = 10

data_new2 = data_new.copy()
for n, im in enumerate(data_new2):
    for k, txt in enumerate(im):
        data_new2[n][k].replace('\t', 'NA')

data_conv = []
for id_t in range(max_len):
    # data_conv.append(data_new[0][id_t] + data_new[1][id_t] + data_new[2][id_t] + data_new[3][id_t] + data_new[4][id_t] + data_new[5][id_t])
    line = data_new[0][id_t] + ' ' + data_new[1][id_t] + ' ' + data_new[2][id_t] + ' ' + data_new[3][id_t] + ' ' + data_new[4][id_t] + ' ' + data_new[5][id_t]
    print(line.encode('unicode_escape'))
    # print('%s %s %s %s %s' % (data_new[0][id_t], data_new[1][id_t], data_new[2][id_t], data_new[3][id_t], data_new[4][id_t], data_new[5][id_t]))
    # print('{} "" {} "" {} "" {} "" {}'.format(data_new[0][id_t], data_new[1][id_t], data_new[2][id_t], data_new[3][id_t], data_new[4][id_t], data_new[5][id_t]))
        # data_new[0][id_t] + data_new[1][id_t] + data_new[2][id_t] + data_new[3][id_t] + data_new[4][id_t] + data_new[5][
        #     id_t])

# %%
#
#     seq_found = find_pattern(x, pattern)
#     if seq_found:
#         for token in seq_found.split('\t'):
#             try:
#                 print('%s %s %s %s %s %s' % pattern, end=' ')
#                 coord_fid[id_fid].append(float(token))
#             except ValueError:
#                 pass
#     else:
#         pass



# tab1 = '\t\t\tTime\tStim.\tInter\tFirst\tSecond\tTarget\tRep.\tPeeling\tUser\tCoil\t\t\tCoil\t\t\tCoil\t\t\tEF max.\t\t\tEF max.\tEF at \tCh1 \tCh1 \tCh2 \tCh2 \tCh3 \tCh3 \tCh4 \tCh4 \tCh5 \tCh5 \tCh6 \tCh6\n'
# tab2 = '\t\tID\t(ms)\tType\tpulse\tIntens.\tIntens.\tID\tStim.\tDepth\tResp.\tLoc.\t\t\tNormal\t\t\tDir.\t\t\tLoc.\t\t\tValue\tTarget\tAmp. \tLat. \tAmp. \tLat. \tAmp. \tLat. \tAmp. \tLat. \tAmp. \tLat. \tAmp. \tLat.\n'
# tab3 = '\t\t\t\t\tInt.\t(%)\t(%)\t\tID\t(%)\t\tx\ty\tz\tx\ty\tz\tx\ty\tz\tx\ty\tz\t(V/m)\t(V/m)\t(ÂµV)\t(ms)\t(ÂµV)\t(ms)\t(ÂµV)\t(ms)\t(ÂµV)\t(ms)\t(ÂµV)\t(ms)\t(ÂµV)\t(ms)\n'
