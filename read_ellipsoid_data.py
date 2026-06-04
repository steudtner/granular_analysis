#!/usr/bin/env python3
"""
Tool for parsing ellipsoid binary data.
"""
__author__ = "Ricardo Rebel"
__copyright__ = "Copyright 2024+, Chair of Fluid Mechanics, TU Dresden, Germnany"
__maintainer__ = "Ricardo Rebel"
__email__ = "ricardo.rebel@tu-dresden.de"
__status__ = "Production"


def cli():

    import argparse
    from importlib import metadata
    import sys
        
    try:
        metadata.version("scipy")
    except metadata.PackageNotFoundError:
        print("\n Required package 'scipy' is not installed.")
        print(" Run: pip install scipy\n")
        sys.exit(1)

    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter )

    parser.add_argument('--data', type=str, required=True, help='"trn", "frc", or "coll"')
    parser.add_argument('--counter', type=int, required=True, help='ell_counter for filename')
    parser.add_argument('--path', type=str, required=True, help='path to result files')
    parser.add_argument('--dry', action='store_true', help='dry simulation w/o fluid (default: False)')
    
    args = parser.parse_args()

    ellipsoid_data = load_data(**vars(args))

    print("\n Example:\n")
    print(ellipsoid_data[0])

    # ***********************************************************************************************************
    # ellipsoid_data now contains the desired data
    # ***********************************************************************************************************
    #
    # if data == 'trn'  ---> ellipsoid_data[n] 
    #                        contains transient data for n-th ellipsoid
    #
    # if data == 'frc'  ---> ellipsoid_data[rk_step][n] 
    #                        contains forces/impulses acting on n-th ellipsoid during rk_step
    #
    # if data == 'coll' ---> ellipsoid_data[rk_step][n] 
    #                        contains collision impulses and contact points of n-th collision pair during rk_step
    #
    # ***********************************************************************************************************

    return None


def load_data(**kwargs):
    import sys

    fname = f"{kwargs['path']}/ellipsoid_{kwargs['data']}/ell_{kwargs['data']}_{kwargs['counter']:06d}.bin"
    fname_rk1 = f"{kwargs['path']}/ellipsoid_{kwargs['data']}/ell_{kwargs['data']}_{kwargs['counter']:06d}_1.bin"
    fname_rk2 = f"{kwargs['path']}/ellipsoid_{kwargs['data']}/ell_{kwargs['data']}_{kwargs['counter']:06d}_2.bin"
    fname_rk3 = f"{kwargs['path']}/ellipsoid_{kwargs['data']}/ell_{kwargs['data']}_{kwargs['counter']:06d}_3.bin"

    if kwargs['data'] == "trn":
        ellipsoid_data = read_ellipsoid_trn(fname)
    elif kwargs['data'] == "frc":
        ellipsoid_data_rk1 = read_ellipsoid_frc(fname_rk1)
        ellipsoid_data = ellipsoid_data_rk1
        if not kwargs['dry']:
            ellipsoid_data_rk2 = read_ellipsoid_frc(fname_rk2)
            ellipsoid_data_rk3 = read_ellipsoid_frc(fname_rk3)
            ellipsoid_data = (ellipsoid_data_rk1, ellipsoid_data_rk2, ellipsoid_data_rk3)
    elif kwargs['data'] == "coll":
        ellipsoid_data_rk1 = read_ellipsoid_coll(fname_rk1)
        ellipsoid_data = ellipsoid_data_rk1
        if not kwargs['dry']:
            ellipsoid_data_rk2 = read_ellipsoid_coll(fname_rk2)
            ellipsoid_data_rk3 = read_ellipsoid_coll(fname_rk3)
            ellipsoid_data = (ellipsoid_data_rk1, ellipsoid_data_rk2, ellipsoid_data_rk3)
    else:
        print("\n 'data' must be in ['trn', 'frc', 'coll']")
        sys.exit(1)

    return ellipsoid_data


def print_progress(n, N_tot, data_type, bar_length=40):
    percent = n / N_tot
    filled_length = int(bar_length * percent)
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    print(f'\rLoading {data_type}: |{bar}| {n}/{N_tot} ({percent:.0%})', end='')
    if n == N_tot:
        print("")


# read ell_trn_<ell_counter>.bin
# ell_trn_<ell_counter>.bin contains position, orientation, velocity and shape of ellipsoids
def read_ellipsoid_trn(fname):
    from scipy.io import FortranFile

    # define list of ellipsoids
    ellipsoids_trn = []

    # open binary file
    with FortranFile(fname, 'r') as f:

        # read number of ellipsoids and time
        Nell = f.read_record('i4')[0] # number of ellipsoids
        t    = f.read_record('f8')[0]  # time

        # iterate over every ellipsoid
        for i in range(Nell):

            #print_progress(i+1, Nell, "ellipsoids")

            x  = f.read_record('f8')    # position of center of mass
            u  = f.read_record('f8')    # velocity of center of mass
            q  = f.read_record('f8')    # quaternion (scalar-first order)
            a  = f.read_record('f8')[0] # semi-axis a
            b  = f.read_record('f8')[0] # semi-axis b
            c  = f.read_record('f8')[0] # semi-axes c
            nl = f.read_record('i4')[0] # number of grid points

            # append the ellipsoid to the list of ellipsoids
            ellipsoids_trn.append({
                'x'    : x      ,
                'u'    : u      ,
                'q'    : q      ,
                'a'    : a      ,
                'b'    : b      , 
                'c'    : c      ,
                'nl'   : nl     ,
                't': t
            })

    return ellipsoids_trn


# read ell_frc_<ell_counter>.bin
# ell_frc_<ell_counter>.bin contains forces and moments acting on ellipsoids
def read_ellipsoid_frc(fname):
    from scipy.io import FortranFile
    import re

    # define list of ellipsoids
    ellipsoids_frc = []

    match = re.search(r'_(1|2|3)\.bin$', fname)
    rk_step = int(match.group(1))

    # open binary file
    with FortranFile(fname) as f:

        Nell = f.read_record('i4')[0]    
        t    = f.read_record('f4')[0]
        dt   = f.read_record('f4')[0]

        # iterate over every ellipsoid
        for i in range(Nell):

            print_progress(i+1, Nell, "forces")

            f_f  = f.read_record('f4') # force due to fluid
            f_c  = f.read_record('f4') # force due to collision (includes pw_c, pv_c, pi_c)
            f_b  = f.read_record('f4') # force due to gravity
            m_f  = f.read_record('f4') # torque due to fluid 
            m_c  = f.read_record('f4') # torque due to collision
            pw_c = f.read_record('f4') # total impulse due to external loads (gravity and lubrication during a collision)
            pi_c = f.read_record('f4') # total impulse due to intersection (is not taken into account in particle velocity)
            pv_c = f.read_record('f4') # total impulse due to relative velodity
            lw_c = f.read_record('f4') # total angular impulse due to external loads (gravity and lubrication during a collision)
            li_c = f.read_record('f4') # total angular impulse due to intersection (is not taken into account in particle angular velocity)
            lv_c = f.read_record('f4') # total angular impulse due to relative velodity

            # append the collision to the list of collisions
            ellipsoids_frc.append({
                'rk_step' : rk_step,
                't'       : t,
                'dt'      : dt,
                'f_f'     : f_f,
                'f_c'     : f_c,
                'f_b'     : f_b,
                'm_f'     : m_f,
                'm_c'     : m_c,
                'pw_c'    : pw_c,
                'pi_c'    : pi_c,
                'pv_c'    : pv_c,
                'lw_c'    : lw_c,
                'li_c'    : li_c,
                'lv_c'    : lv_c
            })
            
    return ellipsoids_frc


# read ell_coll_<ell_counter>.bin
# ell_coll_<ell_counter>.bin contains collision impulses acting between collision pairs
def read_ellipsoid_coll(fname):
    from scipy.io import FortranFile
    import re

    # define list of ellipsoids
    ellipsoids_coll = []

    match = re.search(r'_(1|2|3)\.bin$', fname)
    rk_step = int(match.group(1))

    # open binary file
    with FortranFile(fname) as f:

        # read time and number of collisions
        Nc = f.read_record('i4')[0]    
        t  = f.read_record('f4')[0]
        dt = f.read_record('f4')[0]

        # iterate over every collision pair
        for i in range(Nc):

            print_progress(i+1, Nc, "collisions")

            id1  = f.read_record('i4')[0] # id of first particle
            id2  = f.read_record('i4')[0] # id of second particle
            x1   = f.read_record('f4')    # center-of-mass of first particle 
            x2   = f.read_record('f4')    # center-of-mass of second particle
            q1   = f.read_record('f4')    # quaternion of first particle 
            q2   = f.read_record('f4')    # quaternion of second particle 
            cp1  = f.read_record('f4')    # contact point of first particle (in body-fixed system)
            cp2  = f.read_record('f4')    # contact point of second particle (in body-fixed system)
            pw_c = f.read_record('f4')    # impulse due to external loads (gravity and lubrication during a collision)
            pv_c = f.read_record('f4')    # impulse due to relative velodity
            pi_c = f.read_record('f4')    # impulse due to intersection (is not taken into account in particle velocity)

            # append the collision to the list of collisions
            ellipsoids_coll.append({
                'rk_step' : rk_step,
                't'       : t,
                'dt'      : dt,
                'id1'     : id1,
                'id2'     : id2,
                'x1'      : x1,
                'x2'      : x2,
                'q1'      : q1,
                'q2'      : q2,
                'cp1'     : cp1,
                'cp2'     : cp2,
                'pw_c'    : pw_c,
                'pv_c'    : pv_c,
                'pi_c'    : pi_c
            })

    return ellipsoids_coll


# ==============================================================================
# Initiate program
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    cli()
