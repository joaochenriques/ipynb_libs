import h5py
import numpy as np

#~==============================================================================
def save_hdf_array( hdf5_Output, group, name, fdata ):
    hdf5_Output.create_dataset( group + name, data=fdata, 
                                compression="gzip", compression_opts=9  )

def save_hdf_scalar( hdf5_Output, group, name, fdata ):
    hdf5_Output.create_dataset( group + name, data=fdata )

def save_hdf_string( hdf5_Output, group, name, fdata ):
    hdf5_Output.create_dataset( group + name, data=fdata, 
                                dtype=save_hdf_string.dt_str )
save_hdf_string.dt_str = h5py.special_dtype( vlen=bytes )

#~==============================================================================
def load_hdf_array( hdf5_Input, group, name ):
    return np.array( hdf5_Input[ group + name ] )

def load_hdf_scalar( hdf5_Input, group, name ):
    return hdf5_Input[ group + name ][()]

def load_hdf_string( hdf5_Input, group, name ):
    return hdf5_Input[ group + name ][()]
