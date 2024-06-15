from multiprocessing.shared_memory import SharedMemory
import pandas as pd
import numpy as np


class SharedNumpyArray:
    '''
    Wraps a numpy array so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''
    def __init__(self, array, name=None):
        '''
        Creates the shared memory and copies the array therein
        '''
        # create the shared memory location of the same size of the array
        if not name:
            self._shared = SharedMemory(create=True, size=array.nbytes)
        else:
            self._shared = SharedMemory(name=name, size=array.nbytes)
        
        # save data type and shape, necessary to read the data correctly
        self._dtype, self._shape = array.dtype, array.shape
        
        # create a new numpy array that uses the shared memory we created.
        # at first, it is filled with zeros
        res = np.ndarray(
            self._shape, dtype=self._dtype, buffer=self._shared.buf
        )
        
        # copy data from the array to the shared memory. numpy will
        # take care of copying everything in the correct format
        res[:] = array[:]

    def read(self):
        '''
        Reads the array from the shared memory without unnecessary copying.
        '''
        # simply create an array of the correct shape and type,
        # using the shared memory location we created earlier
        return np.ndarray(self._shape, self._dtype, buffer=self._shared.buf)

    def get_name(self):
        return self._shared.name

    def copy(self):
        '''
        Returns a new copy of the array stored in shared memory.
        '''
        return np.copy(self.read())
        
    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._shared.close()
        self._shared.unlink()


class SharedPandasDataFrame:
    '''
    Wraps a pandas dataframe so that it can be shared quickly among processes,
    avoiding unnecessary copying and (de)serializing.
    '''
    def __init__(self, df, name=None):
        '''
        Creates the shared memory and copies the dataframe therein
        '''
        self._values = SharedNumpyArray(df.values, name=name)
        self._index = df.index
        self._columns = df.columns

    def read(self):
        '''
        Reads the dataframe from the shared memory
        without unnecessary copying.
        '''
        return pd.DataFrame(
            self._values.read(),
            index=self._index,
            columns=self._columns
        )
    
    def copy(self):
        '''
        Returns a new copy of the dataframe stored in shared memory.
        '''
        return pd.DataFrame(
            self._values.copy(),
            index=self._index,
            columns=self._columns
        )
        
    def get_name(self):
        return self._values.get_name()

    def unlink(self):
        '''
        Releases the allocated memory. Call when finished using the data,
        or when the data was copied somewhere else.
        '''
        self._values.unlink()
