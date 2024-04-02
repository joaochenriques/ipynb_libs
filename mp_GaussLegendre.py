# Arbitrary Precision Gauss-Legendre Quadrature
#
# The Gauss-Legendre quadrature’s nodes and weights can be calculated from the 
# eigenvalues and eigenvectors of a Jacobi matrix, built from the coefficients 
# of a 3 term recurrence relation of orthogonal Legendre polynomials.
#
# Adapted from: https://www.advanpix.com/documentation/users-manual/#gauss

import sympy as sp

def GaussLegendre_PW( ctx, N ):

    def sorted_enumerate(seq):
        return [ i for (v, i) in sorted((v, i) for (i, v) in enumerate(seq)) ]

    one  = ctx.mpf('1')
    two  = ctx.mpf('2')
    o2   = ctx.mpf('1/2')

    vv = ctx.matrix( list( ctx.mpf( str(2*i) )**-two for i in range(1,N) ) )

    T = ctx.zeros( N, N )

    for i in range(N-1):
        beta = o2 / ctx.sqrt( one - vv[i] )
        T[i,1+i] = beta
        T[1+i,i] = beta
    (D,V) = ctx.eig(T)

    s = sorted_enumerate(D)
    x = []
    w = []

    for i in s:
        x.append( D[i] )
        w.append( two*V[0,i]**two )
        
    return x, w


def GaussLegendre2_PW( ctx, N ):

    h_pnts = int( N/2 )

    tolerance = ctx.mpf('10')**ctx.mpf( -( ctx.mp.dps+1 ) )
    
    z = sp.Symbol('z')
    Legendre_n = sp.legendre( N, z )
    DLegendre_n = sp.diff( Legendre_n, z )

    roots = []

    for i in range( 0, h_pnts ):
        
        x = ctx.cos( mp.pi * ( ctx.mpf(i+1) - ctx.mpf('1/4') ) / ( ctx.mpf(N) + ctx.mpf('1/2') ) )

        error = 10 * tolerance
        iters = 0

        while ( error > tolerance ) and ( iters < 1000 ):
            p  = Legendre_n.subs( z, x ).evalf( ctx.mp.dps )
            dp = DLegendre_n.subs( z, x ).evalf( ctx.mp.dps )
            dx = -p / dp
            x = x + dx
            iters += 1
            error = ctx.fabs( dx )
        
        roots.append( x )
        
    points  = []
    weights = []

    points.extend( roots )
    if N % 2 != 0: points.append( mp.mpf('0') )
    points.extend( roots[::-1] )

    for i in range( 0, h_pnts ):
        points[i] *= mp.mpf(-1)
    
    for x in points:
        dp = DLegendre_n.subs( z, x ).evalf( ctx.mp.dps )
        weights.append( ctx.mpf(2) / ( ( ctx.mpf(1) - x**ctx.mpf(2) ) *( dp**ctx.mpf(2) ) ) )

    return points, weights



def GaussLobbato_PW( n_int ):
    x = None
    w = None
    if n_int == 2:
        w = [mp.mpf("1"), mp.mpf("1")]
        x = [mp.mpf("-1"), mp.mpf("1")]
    if n_int == 3:
        w = [mp.mpf("0.33333333333333333333333333333333333333333333333333"),
             mp.mpf("1.3333333333333333333333333333333333333333333333333"),
             mp.mpf("0.33333333333333333333333333333333333333333333333333")]
        x = [mp.mpf("-1"),mp.mpf("0"),mp.mpf("1")]
    if n_int == 4:
        w = [mp.mpf("0.16666666666666666666666666666666666666666666666667"),
             mp.mpf("0.83333333333333333333333333333333333333333333333333"),
             mp.mpf("0.8333333333333333333333333333333333333333333333333"),
             mp.mpf("0.16666666666666666666666666666666666666666666666667")]
        x = [mp.mpf("-1"),
             mp.mpf("-0.44721359549995793928183473374625524708812367192231"),
             mp.mpf("0.44721359549995793928183473374625524708812367192231"),
             mp.mpf("1")]
    if n_int == 5:
        w = [mp.mpf("0.1"),
            mp.mpf("0.5444444444444444444444444444444444444444444444444"),
            mp.mpf("0.71111111111111111111111111111111111111111111111111"),
            mp.mpf("0.54444444444444444444444444444444444444444444444444"),
            mp.mpf("0.1")]
        x = [mp.mpf("-1"),
             mp.mpf("-0.65465367070797714379829245624685835556920808239543"),
             mp.mpf("0"),
             mp.mpf("0.65465367070797714379829245624685835556920808239543"),
             mp.mpf("1")]
    if n_int == 6:
        w = [mp.mpf("0.066666666666666666666666666666666666666666666666667"),
             mp.mpf("0.3784749562978469803166128082120246524763246938973"),
             mp.mpf("0.55485837703548635301672052512130868085700863943608"),
             mp.mpf("0.55485837703548635301672052512130868085700863943608"),
             mp.mpf("0.37847495629784698031661280821202465247632469389725"),
             mp.mpf("0.066666666666666666666666666666666666666666666666667")]
        x = [mp.mpf("-1"),
             mp.mpf("-0.7650553239294646928510029739593381503657356885361"),
             mp.mpf("-0.28523151648064509631415099404087907191900347272643"),
             mp.mpf("0.28523151648064509631415099404087907191900347272643"),
             mp.mpf("0.7650553239294646928510029739593381503657356885361"),
             mp.mpf("1")]
    if n_int == 7:
        w = [mp.mpf("0.047619047619047619047619047619047619047619047619048"),
             mp.mpf("0.2768260473615659480107004062900662934976272801799"),
             mp.mpf("0.43174538120986262341787102228136227793094414839155"),
             mp.mpf("0.48761904761904761904761904761904761904761904761905"),
             mp.mpf("0.43174538120986262341787102228136227793094414839155"),
             mp.mpf("0.2768260473615659480107004062900662934976272801799"),
             mp.mpf("0.047619047619047619047619047619047619047619047619048")]
        x = [mp.mpf("-1"),
             mp.mpf("-0.830223896278566929872032213967465139587170364872"),
             mp.mpf("-0.46884879347071421380377188190876632940559747167184"),
             mp.mpf("0"),
             mp.mpf("0.46884879347071421380377188190876632940559747167184"),
             mp.mpf("0.830223896278566929872032213967465139587170364872")]
    if n_int == 8:
        w = [mp.mpf("0.035714285714285714285714285714285714285714285714286"),
             mp.mpf("0.21070422714350603938299206577575632445534616616105"),
             mp.mpf("0.3411226924835043647642406771077481717751109756056"),
             mp.mpf("0.41245879465870388156705297140220978948382857251908"),
             mp.mpf("0.4124587946587038815670529714022097894838285725191"),
             mp.mpf("0.34112269248350436476424067710774817177511097560558"),
             mp.mpf("0.21070422714350603938299206577575632445534616616105"),
             mp.mpf("0.035714285714285714285714285714285714285714285714286")]
        x = [mp.mpf("-1"),
             mp.mpf("-0.87174014850960661533744576122066343810378066967698"),
             mp.mpf("-0.59170018143314230214451073139795318994570098951733"),
             mp.mpf("-0.20929921790247886876865726034535125529554540508668"),
             mp.mpf("0.20929921790247886876865726034535125529554540508668"),
             mp.mpf("0.59170018143314230214451073139795318994570098951733"),
             mp.mpf("0.87174014850960661533744576122066343810378066967698"),
             mp.mpf("1")]
    return x,w


    
if __name__ == '__main__':

    import mpmath as mp
    mp.mp.dps=20
    
    x, w = GaussLegendre_PW( mp, 7 )
    for i in range(7):
        print( f'{x[i]} {w[i]}' )