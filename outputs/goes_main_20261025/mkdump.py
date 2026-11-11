import sys
for f,out in [('outputs/q_main_20261017/figs_compare5.py','outputs/goes_main_20261025/figs/pts_skynet.npz'),('outputs/goes_main_20261025/figs_compare4_goes.py','outputs/goes_main_20261025/figs/pts_goes.npz')]:
    s=open(f).read().replace(chr(13),'')
    i=s.index('fig, axs')
    s=s[:i]+'np.savez(%r, GT=GT, names=np.array(names), **{n:p for n,p in zip(names,rows)})'%out+chr(10)
    open(f.replace('.py','_dump.py'),'w').write(s)
    print(f, 'names' in s[:i])