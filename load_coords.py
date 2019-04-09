# %%
#Loading the coil coordinates from NBS eXimia session file
import re

# %%
filename = 's01_eximia_coords.txt'
with open(filename, 'r') as f:
    x = f.readlines()

print(x)

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
id_fid = 0
# print('%s' % pattern, end=' ')
id_line = 0
for s in x:
    seq_found = re.search(pattern, s)
    if seq_found:
        for id_l in range(6):
            data.append(x[id_fid + id_l].split('\t'))
        break
    id_fid += 1

print(data)
# print(x[id_fid])
data_new = data.copy()

# %%

a = []
for im in data_new:
    a.append(len(im))
max_len = max(a)

for n, im in enumerate(data):
    if len(im) < max_len:
        # data_extra = ['' for id_l in range(max_len - len(im))]
        data_new[n].extend([' ' for id_l in range(max_len - len(im))])

for n, i in enumerate(a):
    if i == 1:
        a[n] = 10

data_conv = []
for id_t in range(max_len):
    # data_conv.append(data_new[0][id_t] + data_new[1][id_t] + data_new[2][id_t] + data_new[3][id_t] + data_new[4][id_t] + data_new[5][id_t])
    print(data_new[0][id_t] + ' ' + data_new[1][id_t] + ' ' + data_new[2][id_t] + ' ' + data_new[3][id_t] + ' ' + data_new[4][id_t] + ' ' + data_new[5][id_t])
    # print('%s %s %s %s %s' % (data_new[0][id_t], data_new[1][id_t], data_new[2][id_t], data_new[3][id_t], data_new[4][id_t], data_new[5][id_t]))
    # print('{} "" {} "" {} "" {} "" {}'.format(data_new[0][id_t], data_new[1][id_t], data_new[2][id_t], data_new[3][id_t], data_new[4][id_t], data_new[5][id_t]))
        # data_new[0][id_t] + data_new[1][id_t] + data_new[2][id_t] + data_new[3][id_t] + data_new[4][id_t] + data_new[5][
        #     id_t])

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
