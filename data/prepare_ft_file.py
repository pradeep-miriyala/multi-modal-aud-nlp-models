def write_ft_file(X,y,fname):
    with open(fname, 'w',encoding="utf-8") as f:
        for i,r in X.iteritems():
            f.write(f'__label__{y[i]} {X[i]}')
            f.write('\n')