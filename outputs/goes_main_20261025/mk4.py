s=open('figs_compare5_goes.py').read().replace(chr(13),'')
L=[l for l in s.split(chr(10)) if not l.startswith(('g = np.load(','idx = ','ranerf = '))]
s=chr(10).join(L)
for a,b in [(', ranerf]',']'),(", 'ranerf']",']'),('subplots(5, 3','subplots(4, 3'),('8.6)','6.95)'),('comparison_five_methods_goes.png','comparison_four_methods_goes.png')]:
    assert a in s, a
    s=s.replace(a,b)
open('figs_compare4_goes.py','w').write(s)
print('ok', s.count('ranerf'))