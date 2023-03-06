import numpy as np

#########################################################################################
# Adapted from:
# https://github.com/mhackemack/POLYFEM/blob/master/Codes/MATLAB/Tools/FEM/triquad.m
#########################################################################################

def QuadRuleTriangles( order ):
#
#   Order:
#       An positive integer indicating the desired order of accuracy for the integration
#       method, i.e. (h^Order) accuracy.  Note: not all orders are available ],  only these:
#       order = [1, 3, 4, 6, 7, 9, 12, 13, 19, 28, 37].
#
#   pnts:
#       An N x 2 array, where N is the number of quadrature pnts.  Each row gives the
#       (x,y) coordinates of the point.  Note: N := Order.
#
#   weights:
#       An N x 1 array, where N is the number of quadrature pnts.  Each row gives the
#       weight assigned to the point.  Note: N := Order.  WARNING: the quad weights are
#       scaled such that they sum to (1.0/2) ], this is the area of the reference triangle.
#
    if order == 1: # O(h), degree of precision 0
        pnts    = np.array( [ [ 1.0/3.0, 1.0/3.0] ] )
        weights = np.array( [ 1.0 ] )
            
    elif order == 3: # O(h^3), degree of precision 2
        pnts  = np.array( [ [ 2.0/3.0, 1.0/6.0], 
                            [ 1.0/6.0, 2.0/3.0], 
                            [ 1.0/6.0, 1.0/6.0] ] )
        weights = np.array( [ 1.0/3.0, 1.0/3.0, 1.0/3.0 ] )

    elif order == 4: # O(h^4), degree of precision 3
        pnts  = np.array( [ [ 1.0/3.0, 1.0/3.0], 
                            [ 1.0/5.0, 1.0/5.0], 
                            [ 3.0/5.0, 1.0/5.0], 
                            [ 1.0/5.0, 3.0/5.0] ] )
        weights = np.array( [-27.0/48.0, 25.0/48.0, 25.0/48.0, 25.0/48.0] )

    elif order == 6: # O(h^6), degree of precision 4
        pnts  = np.array( [ [ 0.816847572980459,  0.091576213509771 ],
                            [ 0.091576213509771,  0.816847572980459 ],
                            [ 0.091576213509771,  0.091576213509771 ],
                            [ 0.108103018168070,  0.445948490915965 ],
                            [ 0.445948490915965,  0.108103018168070 ],
                            [ 0.445948490915965,  0.445948490915965 ] ] )
        weights = np.array( [ 0.109951743655322,
                              0.109951743655322,
                              0.109951743655322,
                              0.223381589678011,
                              0.223381589678011,
                              0.223381589678011 ] )
        
    elif order == 7: # O(h^7), degree of precision 5
        pnts  = np.array( [ [ 1./3.,                      1./3. ],                           
                            [ (6.+np.sqrt(15.))    / 21., (6.+np.sqrt(15.))    / 21. ],
                            [ (9.-2.*np.sqrt(15.)) / 21., (6.+np.sqrt(15.))    / 21. ],
                            [ (6.+np.sqrt(15.))    / 21., (9.-2.*np.sqrt(15.)) / 21. ],
                            [ (6.-np.sqrt(15.))    / 21., (6.-np.sqrt(15.))    / 21. ],
                            [ (9.+2.*np.sqrt(15.)) / 21., (6.-np.sqrt(15.))    / 21. ],
                            [ (6.-np.sqrt(15.))    / 21., (9.+2.*np.sqrt(15.)) / 21. ] ] ) 
        weights = np.array( [ 9./40.,
                              (155.+np.sqrt(15.))/1200.,
                              (155.+np.sqrt(15.))/1200.,
                              (155.+np.sqrt(15.))/1200.,
                              (155.-np.sqrt(15.))/1200.,
                              (155.-np.sqrt(15.))/1200.,
                              (155.-np.sqrt(15.))/1200.] )
        
    elif order == 9: # O(h^9), degree of precision 6
        pnts  = np.array( [ [ 0.124949503233232,  0.437525248383384 ],
                            [ 0.437525248383384,  0.124949503233232 ],
                            [ 0.437525248383384,  0.437525248383384 ],
                            [ 0.797112651860071,  0.165409927389841 ],
                            [ 0.797112651860071,  0.037477420750088 ],
                            [ 0.165409927389841,  0.797112651860071 ],
                            [ 0.165409927389841,  0.037477420750088 ],
                            [ 0.037477420750088,  0.797112651860071 ],
                            [ 0.037477420750088,  0.165409927389841 ] ] )
        weights = np.array( [ 0.205950504760887,
                              0.205950504760887,
                              0.205950504760887,
                              0.063691414286223,
                              0.063691414286223,
                              0.063691414286223,
                              0.063691414286223,
                              0.063691414286223,
                              0.063691414286223 ] )
        
    elif order == 12: # O(h^12), degree of precision 6
        pnts  = np.array( [ [ 0.873821971016996,  0.063089014491502 ],
                            [ 0.063089014491502,  0.873821971016996 ],
                            [ 0.063089014491502,  0.063089014491502 ],
                            [ 0.501426509658179,  0.249286745170910 ],
                            [ 0.249286745170910,  0.501426509658179 ],
                            [ 0.249286745170910,  0.249286745170910 ],
                            [ 0.636502499121399,  0.310352451033785 ],
                            [ 0.636502499121399,  0.053145049844816 ],
                            [ 0.310352451033785,  0.636502499121399 ],
                            [ 0.310352451033785,  0.053145049844816 ],
                            [ 0.053145049844816,  0.636502499121399 ],
                            [ 0.053145049844816,  0.310352451033785 ] ] )
        
        weights = np.array( [ 0.050844906370207,
                              0.050844906370207,
                              0.050844906370207,
                              0.116786275726379,
                              0.116786275726379,
                              0.116786275726379,
                              0.082851075618374,
                              0.082851075618374,
                              0.082851075618374,
                              0.082851075618374,
                              0.082851075618374,
                              0.082851075618374 ] )
        
    elif order == 13: # O(h^13), degree of precision 7
        pnts  = np.array( [ [ 0.333333333333333,  0.333333333333333 ],
                            [ 0.479308067841923,  0.260345966079038 ],
                            [ 0.260345966079038,  0.479308067841923 ],
                            [ 0.260345966079038,  0.260345966079038 ],
                            [ 0.869739794195568,  0.065130102902216 ],
                            [ 0.065130102902216,  0.869739794195568 ],
                            [ 0.065130102902216,  0.065130102902216 ],
                            [ 0.638444188569809,  0.312865496004875 ],
                            [ 0.638444188569809,  0.048690315425316 ],
                            [ 0.312865496004875,  0.638444188569809 ],
                            [ 0.312865496004875,  0.048690315425316 ],
                            [ 0.048690315425316,  0.638444188569809 ],
                            [ 0.048690315425316,  0.312865496004875 ] ] )
        weights = np.array( [-0.149570044467670,
                              0.175615257433204,
                              0.175615257433204,
                              0.175615257433204,
                              0.053347235608839,
                              0.053347235608839,
                              0.053347235608839,
                              0.077113760890257,
                              0.077113760890257,
                              0.077113760890257,
                              0.077113760890257,
                              0.077113760890257,
                              0.077113760890257 ] )
        
    elif order == 19: # O(h^19), degree of precision 9
        pnts  = np.array( [ [ 0.3333333333333333,  0.3333333333333333 ],
                            [ 0.0206349616025259,  0.4896825191987370 ],
                            [ 0.4896825191987370,  0.0206349616025259 ],
                            [ 0.4896825191987370,  0.4896825191987370 ],
                            [ 0.1258208170141290,  0.4370895914929355 ],
                            [ 0.4370895914929355,  0.1258208170141290 ],
                            [ 0.4370895914929355,  0.4370895914929355 ],
                            [ 0.6235929287619356,  0.1882035356190322 ],
                            [ 0.1882035356190322,  0.6235929287619356 ],
                            [ 0.1882035356190322,  0.1882035356190322 ],
                            [ 0.9105409732110941,  0.0447295133944530 ],
                            [ 0.0447295133944530,  0.9105409732110941 ],
                            [ 0.0447295133944530,  0.0447295133944530 ],
                            [ 0.7411985987844980,  0.0368384120547363 ],
                            [ 0.7411985987844980,  0.2219628891607657 ],
                            [ 0.0368384120547363,  0.7411985987844980 ],
                            [ 0.0368384120547363,  0.2219628891607657 ],
                            [ 0.2219628891607657,  0.7411985987844980 ],
                            [ 0.2219628891607657,  0.0368384120547363 ] ] )
        weights = np.array( [ 0.09713579628279610,
                              0.03133470022713983,
                              0.03133470022713983,
                              0.03133470022713983,
                              0.07782754100477543,
                              0.07782754100477543,
                              0.07782754100477543,
                              0.07964773892720910,
                              0.07964773892720910,
                              0.07964773892720910,
                              0.02557767565869810,
                              0.02557767565869810,
                              0.02557767565869810,
                              0.04328353937728940,
                              0.04328353937728940,
                              0.04328353937728940,
                              0.04328353937728940,
                              0.04328353937728940,
                              0.04328353937728940 ] )

    elif order == 28: # O(h^28), degree of precision 11
        pnts  = np.array( [ [ 0.33333333333333333,  0.333333333333333333 ],
                            [ 0.94802171814342330,  0.025989140928288330 ],
                            [ 0.02598914092828833,  0.948021718143423300 ],
                            [ 0.02598914092828833,  0.025989140928288330 ],
                            [ 0.81142499470415460,  0.094287502647922700 ],
                            [ 0.09428750264792270,  0.811424994704154600 ],
                            [ 0.09428750264792270,  0.094287502647922700 ],
                            [ 0.01072644996557060,  0.494636775017214700 ],
                            [ 0.49463677501721470,  0.010726449965570600 ],
                            [ 0.49463677501721470,  0.494636775017214700 ],
                            [ 0.58531323477097150,  0.207343382614514200 ],
                            [ 0.20734338261451420,  0.585313234770971500 ],
                            [ 0.20734338261451420,  0.207343382614514200 ],
                            [ 0.12218438859901870,  0.438907805700490700 ],
                            [ 0.43890780570049070,  0.122184388599018700 ],
                            [ 0.43890780570049070,  0.438907805700490700 ],
                            [ 0.67793765488259020,  0.044841677589130550 ],
                            [ 0.67793765488259020,  0.277220667528279250 ],
                            [ 0.04484167758913055,  0.677937654882590200 ],
                            [ 0.04484167758913055,  0.277220667528279250 ],
                            [ 0.27722066752827925,  0.677937654882590200 ],
                            [ 0.27722066752827925,  0.044841677589130550 ],
                            [ 0.85887028128263640,  0.000000000000000000 ],
                            [ 0.85887028128263640,  0.141129718717363600 ],
                            [ 0.00000000000000000,  0.858870281282636400 ],
                            [ 0.00000000000000000,  0.141129718717363600 ],
                            [ 0.14112971871736360,  0.858870281282636400 ],
                            [ 0.14112971871736360,  0.000000000000000000 ] ] )
        weights = np.array( [ 0.087977301162221900,
                              0.008744311553736190,
                              0.008744311553736190,
                              0.008744311553736190,
                              0.038081571993935330,
                              0.038081571993935330,
                              0.038081571993935330,
                              0.018855448056131250,
                              0.018855448056131250,
                              0.018855448056131250,
                              0.072159697544741000,
                              0.072159697544741000,
                              0.072159697544741000,
                              0.069329138705537200,
                              0.069329138705537200,
                              0.069329138705537200,
                              0.041056315429288600,
                              0.041056315429288600,
                              0.041056315429288600,
                              0.041056315429288600,
                              0.041056315429288600,
                              0.041056315429288600,
                              0.007362383783300573,
                              0.007362383783300573,
                              0.007362383783300573,
                              0.007362383783300573,
                              0.007362383783300573,
                              0.007362383783300573 ] ) 
        
    elif order == 37: # O(h^37), degree of precision 13
        pnts  = np.array( [ [ 0.333333333333333333333333333333,  0.333333333333333333333333333333 ],
                            [ 0.950275662924105565450352089520,  0.024862168537947217274823955239 ],
                            [ 0.024862168537947217274823955239,  0.950275662924105565450352089520 ],
                            [ 0.024862168537947217274823955239,  0.024862168537947217274823955239 ],
                            [ 0.171614914923835347556304795551,  0.414192542538082326221847602214 ],
                            [ 0.414192542538082326221847602214,  0.171614914923835347556304795551 ],
                            [ 0.414192542538082326221847602214,  0.414192542538082326221847602214 ],
                            [ 0.539412243677190440263092985511,  0.230293878161404779868453507244 ],
                            [ 0.230293878161404779868453507244,  0.539412243677190440263092985511 ],
                            [ 0.230293878161404779868453507244,  0.230293878161404779868453507244 ],
                            [ 0.772160036676532561750285570113,  0.113919981661733719124857214943 ],
                            [ 0.113919981661733719124857214943,  0.772160036676532561750285570113 ],
                            [ 0.113919981661733719124857214943,  0.113919981661733719124857214943 ],
                            [ 0.009085399949835353883572964740,  0.495457300025082323058213517632 ],
                            [ 0.495457300025082323058213517632,  0.009085399949835353883572964740 ],
                            [ 0.495457300025082323058213517632,  0.495457300025082323058213517632 ],
                            [ 0.062277290305886993497083640527,  0.468861354847056503251458179727 ],
                            [ 0.468861354847056503251458179727,  0.062277290305886993497083640527 ],
                            [ 0.468861354847056503251458179727,  0.468861354847056503251458179727 ],
                            [ 0.022076289653624405142446876931,  0.851306504174348550389457672223 ],
                            [ 0.022076289653624405142446876931,  0.126617206172027096933163647918 ],
                            [ 0.851306504174348550389457672223,  0.022076289653624405142446876931 ],
                            [ 0.851306504174348550389457672223,  0.126617206172027096933163647918 ],
                            [ 0.126617206172027096933163647918,  0.022076289653624405142446876931 ],
                            [ 0.126617206172027096933163647918,  0.851306504174348550389457672223 ],
                            [ 0.018620522802520968955913511549,  0.689441970728591295496647976487 ],
                            [ 0.018620522802520968955913511549,  0.291937506468887771754472382212 ],
                            [ 0.689441970728591295496647976487,  0.018620522802520968955913511549 ],
                            [ 0.689441970728591295496647976487,  0.291937506468887771754472382212 ],
                            [ 0.291937506468887771754472382212,  0.018620522802520968955913511549 ],
                            [ 0.291937506468887771754472382212,  0.689441970728591295496647976487 ],
                            [ 0.096506481292159228736516560903,  0.635867859433872768286976979827 ],
                            [ 0.096506481292159228736516560903,  0.267625659273967961282458816185 ],
                            [ 0.635867859433872768286976979827,  0.096506481292159228736516560903 ],
                            [ 0.635867859433872768286976979827,  0.267625659273967961282458816185 ],
                            [ 0.267625659273967961282458816185,  0.096506481292159228736516560903 ],
                            [ 0.267625659273967961282458816185,  0.635867859433872768286976979827 ] ] )
        weights = np.array( [ 0.051739766065744133555179145422,
                              0.008007799555564801597804123460,
                              0.008007799555564801597804123460,
                              0.008007799555564801597804123460,
                              0.046868898981821644823226732071,
                              0.046868898981821644823226732071,
                              0.046868898981821644823226732071,
                              0.046590940183976487960361770070,
                              0.046590940183976487960361770070,
                              0.046590940183976487960361770070,
                              0.031016943313796381407646220131,
                              0.031016943313796381407646220131,
                              0.031016943313796381407646220131,
                              0.010791612736631273623178240136,
                              0.010791612736631273623178240136,
                              0.010791612736631273623178240136,
                              0.032195534242431618819414482205,
                              0.032195534242431618819414482205,
                              0.032195534242431618819414482205,
                              0.015445834210701583817692900053,
                              0.015445834210701583817692900053,
                              0.015445834210701583817692900053,
                              0.015445834210701583817692900053,
                              0.015445834210701583817692900053,
                              0.015445834210701583817692900053,
                              0.017822989923178661888748319485,
                              0.017822989923178661888748319485,
                              0.017822989923178661888748319485,
                              0.017822989923178661888748319485,
                              0.017822989923178661888748319485,
                              0.017822989923178661888748319485,
                              0.037038683681384627918546472190,
                              0.037038683681384627918546472190,
                              0.037038683681384627918546472190,
                              0.037038683681384627918546472190,
                              0.037038683681384627918546472190,
                              0.037038683681384627918546472190 ] ) 

    else:
        raise Exception('Quadrature rule not chosen!')

    # transform to the used reference element
    # (-1,-1), (+1,-1), (-1,+1) => Area = 2
    px = 2.0*pnts.T[0]-1.0
    py = 2.0*pnts.T[1]-1.0
    ww = 2.0 * weights
    return px, py, ww