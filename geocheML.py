# -*- coding: utf-8 -*-
"""
A class meant to simplify the clustering and plotting of LA-ICP-MS geochemical
data. The initial dataset was mica chemistry from the Brazil Lake Pegmatite,
Yarmouth County, Nova Scotia, Canada.

Version 1.0
2023/03/27
Adam Nissen
"""
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from plotly.offline import plot
from matplotlib import pyplot as plt
import os




class Kmeans_interrogation:
    #Class Variables
    __element_symbols = ('H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O',
                              'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S',
                              'Cl','Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr',
                              'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga',
                              'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr',
                              'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh',
                              'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te',
                              'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr',
                              'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
                              'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta',
                              'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                              'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr',
                              'Ra', 'Ac', 'Th', 'Pa', 'U')
    
    def __init__(self):
        self._folderpath = None
        self._filename = None
        self._raw = None
        self._working = None
        self._excluded_data = None
        self._instance_marker = None
        self._nan_removals = None
        self._zeros = None
        self.cluster_counts = None
        self._subset = None
        
    def load(self, data, instance_marker = None):
        """
        Load ICPMS data into CLASS for later kmeans interrogation and data plotting.
        This will populate two instance variables; self._raw, which is a copy or
        or near copy of the loaded data; and self._working, which only contains
        an instance marker column and abundance data for each element.
        
        Please note that if no valid instance_marker argument is passed, then both
        self._raw and self._working will contain a column labeled 'instance' with
        sequential unique integer values for each row starting at 0. This will 
        overwrite data if your dataframe already has a column named instance.
        
        Parameters
        ----------
        data : pandas DataFrame
            ICPMS data in a pandas DataFrame. Rows should be measurement instances,
            columns should be information channels. This function looks for columns
            with element symbols (e.g. K, Na, Fe, Cs, etc.) in the header, please 
            avoid capitalizing anything but element symbols in the column headers
            to avoid bugs (e.g. 'analysis' is better than 'Analysis' and 'help'
            is far better than 'Help').
        instance_marker : str, column header, optional
            Column header for an instance marking column. This column will be used
            to map cluster numbers to individual measurements, so there must not
            be any repeating values in this column. Analysis numbers often make
            valid instance markers. PLEASE NOTE if no instance_marker is passed,
            or if an invalid instance marker is passed (e.g. if the column header 
            is mis-spelled or if the column contains repeat values), this function
            will create a column called 'instance' filled with non repeating numbers
            ranging from 0 to n, where n is the number of rows in the data argument.
        """
        #Local variables to remove non-data columns from the working dataframe
        drop_labels = []
        valid_marker = False
        
        #Nested for loops evaluating if a column should be kept or dropped
        for column in data.columns:
            keep = False
            #Verifying a column involves an element symbol
            for element in self.__element_symbols:
                if element in column:
                    keep = True
                else:
                    pass
            #Removing error, count, and limit of detection columns
            if '_2SE' in column:
                keep = False
            elif '_n' in column:
                keep = False
            elif '_LODH' in column:
                keep = False    
            #retaining the instance_marker, whether it was called or created by this method
            elif column == instance_marker:
                #checking to ensure each instance in the instance_marker 
                if len(data[column].unique()) == len(data[column]):
                    keep = True
                    valid_marker = True
                else:
                    print('Instance marker not fully unique, generating instance column')
            else:
                pass
            #Filling out the drop_labels list
            if keep == False:
                drop_labels.append(column)
            else:
                pass

        #Adding instance_markers if a valid marker was not present to help map clusters to measurements
        if valid_marker == False:
            data['instance'] = range(len(data))
            self._raw = data
        else:
            self._raw = data
            #Renaming the instance_marker column 'instance' in the working data for simplicity
            data = data.rename(columns = {instance_marker:'instance'})
        
        #Retaining the raw/working data in instance variables 
        
        working = data.drop(columns = drop_labels, axis = 1)
        for column in working.columns:
            if column != 'instance':
                working[column] = pd.to_numeric(working[column], errors = 'coerce')
                if '_mean' in column:    
                    working = working.rename(columns = {column:column.replace('_mean', '')})
            else:
                pass
        self._working = working
        self._zeros = working.fillna(value = 0)
        self._instance_marker = instance_marker

    def remove_nan(self, nan_threshold = None):
        """
        Removes columns with more than n instances of nan (e.g. below LOD measurements)
        from the dataframe, where n is the number input into nan_threshold. Then
        it removes all the remaining rows with nan values in them. 
        
        If no value for nan_threshold is passed the method will return the "nan 
        implications" of the working data. This includes; the lowest nan_threshold 
        that is required to keep each column in the working data; the number of 
        elements (i.e. columns) that will remain in the working data at each 
        nan_threshold; and the number of measurements (i.e. rows) that will remain
        in the working data at each nan_threshold.
        
        Machine learning algorithms often cannot manage nan values. Instead, we 
        must decide how to remove instances without a value. In some cases it may
        be appropreate to replace all nan values with a constant (e.g. 0), however
        it may be more appropreate to remove the nan values by either removing 
        the feature (i.e. column, element) or the instance (i.e. row, sample, analysis)
        from the machine learning algorithm. When removing nans in this way there
        is a trade off between including features, and excluding analysis. Using
        this algorithm, the higher the nan_threshold, the more features or elements
        will be considered by the clustering algorithm. Conversely, higher nan_thresholds
        and higher feature counts will lower the number of available instances 
        for the algorithm.
        
        I recommend finding the lowest nan_threshold that can be selected that
        still includes the most significant features for classification.

        Parameters
        ----------
        nan_threshold : int, range 0 to n, where n is the number of instances or
                        or rows in the working dataframe, optional
            The number of nan values that will be accepted in each column of the 
            working dataframe. Must be greater than -1, and less than len(self._working)
            Lower nan_thresholds will allow the machine learning algorithm to cluster
            more points of data (instances), while higher nan_thresholds will provide
            more features (elements) for the machine learning algorithm to cluster 
            points by. 

        EXAMPLE NAN IMPLICATIONS
        -------
        Si29        42
        Tm169       41
        Lu175       40
        Yb172       40
        Ag107       39
        Dy163       39
        Er166       39
        Tb159       38
        Ho165       37
        Sm147       36
        Gd157       36
        Pr141       17
        Nd146       17
        Th232       17
        U238        11
        Cu65         9
        Ca43         8
        Eu153        8
        Y89          7
        La139        6
        V51          5
        V50          5
        Hf178        5
        Zr91         5
        Nb93         5
        Cs133        5
        W182         4
        W184         4
        Sn118        4
        Ni62         4
        Ce140        3
        Ta181        3
        Cr52         3
        Co59         3
        Mn55         2
        Zn66         2
        Rb85         2
        Li7          1
        Fe57         0
        Ba137        0
        Sr88         0
        Ti48         0
        K39          0
        Al27         0
        Pb208        0
        Na23         0
        Analysis     0
        dtype: int64

        nan_threshold: 42, elements: 46, measurements: 0
        nan_threshold: 41, elements: 45, measurements: 0
        nan_threshold: 40, elements: 44, measurements: 0
        nan_threshold: 39, elements: 42, measurements: 0
        nan_threshold: 38, elements: 39, measurements: 2
        nan_threshold: 37, elements: 38, measurements: 2
        nan_threshold: 36, elements: 37, measurements: 4
        nan_threshold: 17, elements: 35, measurements: 10
        nan_threshold: 11, elements: 32, measurements: 20
        nan_threshold: 9, elements: 31, measurements: 22
        nan_threshold: 8, elements: 30, measurements: 22
        nan_threshold: 7, elements: 28, measurements: 32
        nan_threshold: 6, elements: 27, measurements: 33
        nan_threshold: 5, elements: 26, measurements: 36
        nan_threshold: 4, elements: 20, measurements: 36
        nan_threshold: 3, elements: 16, measurements: 36
        nan_threshold: 2, elements: 12, measurements: 39
        nan_threshold: 1, elements: 9, measurements: 41
        nan_threshold: 0, elements: 8, measurements: 42
        
        In the example dataset, there are 42 possible instances for classification,
        and 46 possible features to classify the instances by. Of the features,
        eight have 0 nan values in them (Analysis, Na23, Pb208, Al27, K39, Ti48,
        Sr88, Ba137, and Fe57). Setting the nan_threshold to 0 will produce a 
        working datasheet of all 42 rows, and only those eight columns. 
        
        Raising the nan_threshold to 2 would increase the number of features to
        include Mn55, Zn66, Rb85, and Li7, along with the eight features that have
        zero nan values in them. This nan_threshold however will lower the available
        number of instances or points to classify down to 39. If these elements
        were key to understanding the distribution of clusters, then losing the 
        three instances is likely worth the tradeoff.
        
        Note that the number of measurements available at a nan_threshold of 3
        (36 instances) matches the number of measurements available at a 
        nan_threshold of 5. In this case, setting a nan_threshold of 3 or 4 would
        remove four to ten features from the clustering algorithm, without providing
        any more instances for clustering. 
        
        Arbitrarily high nan_thresholds can dramatically reduce the number of 
        features available for clustering. In this example any nan_threshold above
        30 would reduce the pool of instances for classification to four or fewer 
        which would not be appropreate for any machine learning algorithms. With 
        these data, any nan_threshold above 7 would nearly halve the available
        instances which likely dramatically reduces the utility of machine learning.
        """
        #If no threshold is offered, this finds implications of each potential threshold (i.e. which columns and rows it will cause to be dropped)
        if nan_threshold == None:
            
            #Finding the number of nans for each element
            col_nul = self._working.isnull().sum(axis = 0).sort_values(ascending = False)
            print('Lowest threshold to include element')
            print(col_nul)
            print('')
            
            #For each unique value of nans per element, finding the columns that would be left at that threshold
            for i in col_nul.unique():
                temp = self._working.dropna(axis = 1, thresh = len(self._working)-i)
                
                #Removing rows containing nans that were not removed due to the threshold 
                temp = temp.dropna(axis = 0)
                print('nan_threshold: '+str(i)+', elements: '+str(len(temp.columns)-1)+', measurements: '+str(len(temp)))
            
            #requesting an input of the threshold. If the input isn't an integer, it doesn't work.
            print('')
            suggested_t = int(input('Please select a nan_threshold: '))
            
            #Verifying the threshold is in a valid range (i.e. is within the number of rows in the datasheet)
            while suggested_t <0 or suggested_t > len(self._working):
                suggested_t = int(input('nan_threshold must be beteen 0 and '+str(len(self._working))+'. Please select a nan_threshold:'))
            
            #setting the suggested threshold to the actual threshold after verification of its integrity
            nan_threshold = suggested_t
        else:
            pass
        
        #Removing columns with more nan values than the threshold
        #note this is different to dropna for simplicity of understanding, dropna
        #requests the number of non nan values that the iterable can have, as 
        #opposed to the number of nan values that it can have
        working = self._working.dropna(axis = 1, thresh = len(self._working)-nan_threshold)
        #Dropping all rows with nan values
        working = working.dropna(axis = 0)
        
        #finding the instances that were removed above
        filtered_instances = list(working['instance'])
        all_instances = list(self._working['instance'])
        removed_instances = []
        for i in all_instances:
            if i in filtered_instances:
                pass
            else:
                removed_instances.append(i)
        #Saving the working dataframe, and the removed nans           
        self._working = working
        self._nan_removals = removed_instances

    def set_folderpath(self, path):
        """
        Defines the folder in which all figures produced by this class will be 
        saved. Path must be a full folderpath. Please note that when copying 
        a file or folderpath using right click "copy as path" (Ctrl+shift+c) the
        pasted path will use "\" to separate folder/directory levels. For python
        to understand the path, these need to be changed to "/", e.g.
        
        "C:/users/main/documents/data/output/"
        
        You may include a slash after the final folder name or not, it will be 
        included regardless

        Parameters
        ----------
        path : str, e.g. "C:/users/main/documents/data/output/"
            The path to the output repository/folder for all of the figures from
            this class.
        """
        if not os.path.exists(path):
            os.makedirs(path)
        if path[-1] == '/':
            pass
        else:
            path = path+'/'
        self._folderpath = path
    
    def set_filename(self, filename):
        """
        Defines a prefix for each figure file produced by this class. Do not include
        a file suffix (e.g. .txt, .csv, .xlsx, etc.). Any figures will use the
        filename as a prefix, e.g.:
            if filename is entered as:
                20220106_trace_elements
            a pairplot figure will be saved as:
                20220106_trace_elements_pariplots.jpg
            a KDE distribution figure will be saved as:
                20220106_trace_elements_distributions.jpg
                etc.

        Parameters
        ----------
        filename : str
            A prefix for any images that will be saved by CLASS.
        """
        if '.' in filename:
            split = filename.split('.')
            filename = split[0]
        else:
            pass
        self._filename = filename
    
    def set_filepath(self, filepath):
        """
        Defines the folder in which all figures produced by this class will be 
        saved, as well as an identifying filename prefix for each figure. Path 
        must be a full filepath, though a file suffix (e.g. .txt, .csv, .xlsx
        etc.)is not necessary. Please note that when copying a filepath using 
        right click "copy as path" (Ctrl+shift+c) the pasted path will use "\" to 
        separate folder/directory levels. For python to understand the path, these
        need to be changed to "/", e.g.
        
        "C:/users/main/documents/data/output/my_sample"
        
        In the above example, figures will use "my_sample" as a prefix for all 
        saved files, for example:
            pariplots would be saved as:
                C:/users/main/documents/data/output/my_sample_pariplot.jpg
            distribution plots would be saved as:
                C:/users/main/documents/data/output/my_sample_distributions.jpg
            etc.

        Parameters
        ----------
        filepath : str, e.g. "C:/users/main/documents/data/output/my_sample"
            The path to the output repository/folder for all of the figures from
            this class with a fileprefix after the last slash.
        """
       #removing any file suffixes 
        if '.' in filepath:
            split = filepath.split('.')
            filepath = split[0]
        else:
            pass
        
        #Defining path rules for a folder without file prefixes
        if filepath[-1] == '/':
            self._filename = None
            if not os.path.exists(filepath):
                os.makedirs(filepath)
            self._folderpath = filepath
        #Defining path rules for a path with file prefixes
        else:
            folders = filepath.split('/')
            self._filename = folders[-1]
            
            #removing the filename from the folderpath by subtracting the length 
            #of the filename from the filepath. Dropping, removing, or replacing
            #the name causes issues if folders use repeated verbiage 
            filename_length = len(folders[-1])
            folderpath = filepath[:-filename_length]
            
            #Building the path
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)
            
            self._folderpath = folderpath

    def cluster(self, clusters = 3, exclude_elements = None, exclude_instances = None, include_elements = None):
        """
        This method finalizes data processing for, and then applies, a KMeans 
        clustering algorithm to the working data. Labels from the KMeans clustering
        are applied to the working and raw data. A new attribute called 
        self.cluster_counts is created which lists the number of instances in
        each cluster. This is meant to be used as a proxy for how effective the
        clustering is. The cluster counts will be output in a dictionary, e.g.:
            
            {0: 23, 1: 12, 2: 2}
        
        In the example cluster count, cluster 0 has 23 instances, cluster 1 has
        12 instances, and cluster 2 has 2 instances. This is meant to be a simple 
        proxy for how effective the clustering has been. If a cluster count involves
        lots of clusters with a single instance, that may indicate that the number
        of clusters requested from the algorithm is too high.
        
        The argument exclude_instances is used to withold data from the fitting
        stage of the clustering. Some instances will create an "arbitrary" cluster,
        i.e. a cluster you are not searching for. In this case it is best to withold
        these data while fitting the clustering algorithm, then predicting the
        labels for these data with the algorithm once it is fit to the included
        data. 
        
        An example of an "arbitrary" cluster in LA-ICPMS data is a subset of points
        that were collected with larger laser spots. In a series of 60 points, 
        if most of them were collected with 55 μm spots and the remaining five
        were collected with 150 μm spots, the five larger spots may aggregate in 
        a cluster of their own even if the sampling medium is the same from as 
        one of the smaller spot clusters. In this case witholding the larger spots
        from the clustering algorithm can produce more meaningful clusters in the 
        end. 

        Parameters
        ----------
        clusters : int, optional
            The number of clusters that the KMeans classifier should be looking 
            for. The default is 3.
        exclude_elements : list, optional
            A list of elements that should be ignored by the KMeans clusterer.
            This list can be the entire column header for the element column, though
            is can also be just the element symbol (e.g. Sr, Cr, Pb). If a list
            is passed for exclude_elements AND include_elements, the include_elements
            list will take priority. If None is passed, no elements will be removed.
            The default is None.
        exclude_instances : list, optional
            A list of instance labels for instances that should be left out of 
            the clustering algorithm. This could be instances that will "arbitrarily"
            form a cluster that you are not looking for. These excluded instances
            will be classified by the clustering algorithm after it is fit to the 
            included data, but they will not be able to affect the shape of the
            clusters. If None is passed, no instances will be removed. The default
            is None.
        include_elements : list, optional
            A list of elements that should be included in the KMeans clustering
            algorithm. The list can be the entire column header, or it can be 
            just the element symbol (e.g. Sr, Cr, Pb). If a list is passed for
            both exclude_elements AND include_elements, the include_elements list
            will take priority. If None is passed, all elements will be included.
            The default is None.
        """
        
        #Building a "training" set of data that will be used for clustering and 
        #a "testing" set that will not be used for clustering, but will be labeled
        train = self._working
        test = pd.DataFrame()
        
        #Removing no elements if both element lists are empty
        if exclude_elements == None and include_elements == None:
            pass
        
        #Removing elements not in include_elements if any are entered
        elif include_elements != None:
            if type(include_elements) == str:
                include_elements = [include_elements]
            else:
                pass        
            for column in train:
                keep = False
                for element in include_elements:
                    if element in column:
                        keep = True
                    else:
                        pass
                if keep == False and column != 'instance':
                    train = train.drop(labels = column, axis = 1)
        
        #Removing elements in exclude_elements if any are entered
        else:
            if type(exclude_elements) == str:
                exclude_elements = [exclude_elements]
            else:
                pass
            for column in train:
                for element in exclude_elements:
                    if element in column:
                        train = train.drop(labels = column, axis = 1)
        
        #Sepparating the training and testing data as requested by exclude_instances
        if exclude_instances == None:
            pass
        else:
            #### NOTE ###########################################################
            # in the below example using the hashed out lines raises a          #
            # SettingWithCopyError. This error is an effort from pandas to      #
            # eliminate recursive errors with dataframe slices. It doesn't apply#
            # here, but the false positive flag will be suppressed by using .loc#
            # like in the lines that are not hashed out                         #
            #                                                                   #
            #             WELL NOW THEY BOTH RAISE THE ERROR - Great            #
            #####################################################################
            
            #test = train.query('instance in @exclude_instances')
            #train = train.query('instance not in @exclude_instances')
            test = train.copy() #Okay, this line suppresses the error
            test = test.loc[train.instance.isin(exclude_instances)]
            train = train.copy() #Same with this line
            train = train.loc[~train.instance.isin(exclude_instances)]
            
        #Saving a list of the elements used in the clustering as self._subset
        subset = list(train.columns)
        self._subset = tuple(subset)
        
        #scaling the two sets for classification, removing the instance column
        scaler = StandardScaler()
        inset = scaler.fit_transform(train.drop('instance', axis = 1))
        exset = None
        
        #Clustering the data
        alg = KMeans(n_clusters = clusters)
        alg.fit(inset)

        #Saving the number of members of each cluster as an attribute 
        unique, counts = np.unique(alg.labels_, return_counts = True)
        self.cluster_counts = dict(zip(unique,counts))
        
        #Applying labels to the train test data and generating a map to label the raw and working data
        train['cluster'] = alg.labels_
        label_map = dict(zip(list(train['instance']), list(train['cluster'])))
        
        #Scaling, classifying and mapping the witheld data (if any) 
        if exclude_instances == None:
            pass
        else:
            exset = scaler.transform(test.drop('instance', axis = 1))
            test['cluster'] = alg.predict(exset)
            label_map.update(dict(zip(list(test['instance']), list(test['cluster']))))
        
        #Attaching the cluster numbers to the working and raw dataset
        self._working['cluster'] = self._working['instance'].map(label_map)
        self._raw['cluster'] = self._raw[self._instance_marker].map(label_map) #remember, self._raw doesn't necesarily have a column named 'instance', so we had to call the instance_marker variable 
        
    def pair_plot(self, subset_list = None, plot_full = False, plot_subset = True, colour_clusters = True, filepath = None, save_plots = False):
        """
        Produces pair plots using the seaborn package (seaborn.pairplot()). Arguments
        permit the plotting of a subset of features, which will be smaller, more
        manageable, and less taxing on a computer, or the whole set of features.
        The features can be plotted wholesale, or aggregated into clusters. 

        Parameters
        ----------
        subset_list : list, optional
            A list of columns in the working data, or element symbols that should
            be plotted as a subset pairplot. If None is passed as an argument,
            the features used by the clustering are selected as the subset.The 
            default is None.
        plot_full : boolean, optional
            Whether or not to plot the full suite of elements in a pairplot. Pass
            True to plota full suite pairplot, otherwise plot False. Please note
            if your data has a high number of features (i.e. n > 10) the pairplot
            will take a long time and a lot of memory to plot. If your data has
            a very high number of features (i.e. n >> 10) setting plot_full to 
            True could exceed the memory limit on your computer. The default is
            False.
        plot_subset : boolean, optional
            Whether or not to plot a defined subset of the features in a pairplot.
            The subset is defined by the argument subset_list. Note that for data
            with large volumes of features plotting a subset can reduce the size 
            of the pairplot to be more manageable for computing. The default is
            True.
        colour_clusters : boolean, optional
            Whether or not to plot clusters in the pairplots. If true the cluster
            numbers will be entered as the argument "hue" when calling 
            seaborn.pairplot(), and the clusters will be plotted as different
            colours in the pairplots. Note that the palette argument is always
            set to "colourblind" when colour_clusters == True. The default is True.
        filepath : string, optional
            The full filepath for the directory in which the pairplots should be 
            saved. If None is entered, this method will use the filepath defined
            by self.filepath or self.folderpath. The default is None.
        save_plots : boolean, optional
            Whether or not to save the pairplots into the default directory. Pass
            True to save the pairplots, otherwise pass False. The default is False.
            Note that if no default directory is set and save_plots == True no 
            figures will be saved.
        """        
        #Defining working variable dataframes for the pairplot types
        working = self._working
        subset = pd.DataFrame()
        
        
        #Filling out the data subset
        if subset_list == None:
            for i in self._subset:
                subset[i] = working[i]
                subset['cluster'] = working['cluster']
        else:
            for column in working.columns:
                for i in subset_list:
                    if i in column:
                        subset[i] = working[column]
                        subset['cluster'] = working['cluster']
                        subset['instance'] = working['instance']
                        
        #Building the specific filepaths, this code chunk can be reused.
        if filepath == None:
            pass
        else:
            self.set_filepath(filepath = filepath)
        #Recording the path and prefix required to            
        if self._folderpath != None and self._filename != None:
            filepath = self._folderpath+self._filename
        elif self._folderpath != None and self._filename == None:
            filepath = self._folderpath
        else:
            pass
        
        #Clearing the current plot before starting a new plot
        #plt.clf()#Hashed out to see if this is important or not. If the plotting messes up, remove the hash
        
        #Setting a string to designate the plot type in its name
        subset_suffix = ''
        #Making the subset pairplots
        if plot_subset == True and colour_clusters == True:
            subset_plot = sns.pairplot(data = subset.drop('instance', axis = 1), hue = 'cluster', diag_kind = 'kde', palette = 'colorblind')
            subset_suffix = '_clustered'
        elif plot_subset == True and colour_clusters == False:
            subset_plot = sns.pairplot(data = subset.drop(['instance', 'cluster'], axis = 1), diag_kind = 'kde')
            if len(subset.columns) > 10:
                subset_plot.map_lower(sns.kdeplot, levels = 4)
            else:
                pass
        else:
            pass
        
        #Saving the subset pair plot
        if save_plots == True and filepath != None and plot_subset == True:
            plt.savefig(filepath+'_subset_pairplot'+subset_suffix+'.jpg', dpi = 300)
        elif save_plots == True and filepath == None:
            print("Please select a valid filepath when calling self.pair_plots()")
        else:
            pass
            
        #Clearing the current plot before starting a new plot        
        #plt.clf()#Hashed out to see if this is important or not. If the plotting messes up, remove the hash
        
        #Setting a string to designate the plot type in its name
        superset_suffix = ''
        #Making the superset pairplots
        if plot_full == True and colour_clusters == True:
            full_plot = sns.pairplot(data = working.drop('instance', axis = 1), hue = 'cluster', diag_kind = 'kde', palette = 'colorblind')
            superset_suffix = '_clustered'
        elif plot_full == True and colour_clusters == False:
            full_plot = sns.pairplot(data = working.drop(['instance', 'cluster'], axis = 1), diag_kind = 'kde')
        else:
            pass

        #Saving the subset pair plot
        if save_plots == True and filepath != None and plot_full == True:
            plt.savefig(filepath+'_full_pairplot'+superset_suffix+'.jpg', dpi = 300)
        else:
            pass        
        

    def distribution_plot(self, colour_clusters = True, title = None, filepath = None, save_plots = False, x_label = 'ppm'):
        """
        Plots KDE functions for each column in a dataframe. Produced figure will have
        n sub figures in a 1 x n grid where n is the number of columns in the dataframe.
        Multiple KDEs can be plotted in each subfigure if colour_clusters == True.
        In this case, each cluster will be plotted on each subfigure in a different
        colour. Note that ten colourblind friendly colours have been coded into
        the plot. If you are calling more than 10 clusters, the plot may fail.
        
        The dotted line plotted in each subfigure is the median of the plotted 
        population.
        
        In the legend, n is the size of each plotted population.

        Parameters
        ----------
        colour_clusters : boolean, optional
            Whether or not to plot clusters as different coloured KDEs in each
            subplot. Pass True to plot clusters, pass False to plot the unclustered
            distributions. The default is True.
        title : string, optional
            Title for the entire plot. If None is passed, no title will be added.
            The default is None.
        filepath : str, optional
            c
            Whether or not to save the distribution plots into the default directory. 
            Pass true to save the pairplots, otherwise pass False. The default 
            is False. Note that if no default directory is set and save_plots 
            == True, no figures will be saved.
        x_label : str, optional
            A super x label that will be set below the x axis of the lowermost
            subplot, meant to describe the units for each of the distribution
            plots. The default is 'ppm'.
        """
        #Grabbing the working data
        working = self._working
        
        #building a specific filepath for the output of this method
        if filepath == None:
            pass
        else:
            self.set_filepath(filepath = filepath)
        #Recording the path and prefix required to            
        if self._folderpath != None and self._filename != None:
            filepath = self._folderpath+self._filename
        elif self._folderpath != None and self._filename == None:
            filepath = self._folderpath
        else:
            pass
        
        #clearing the previous plot to prepare for the distribution plot
        plt.clf()
        
        #Defining the number of subplots in the distribution plot
        rows = working.columns.drop('instance')
        if 'cluster' in rows:
            rows = rows.drop('cluster')
        rows = list(rows)
        
        
        #Setting up the matplotlib figure
        fig, axs = plt.subplots(nrows = len(rows), ncols = 1, figsize = (7, len(rows)))
        
        #Iterating over axes to fill the subfigures
        for i in range(len(rows)):
            ax = axs[i]
            column = rows[i]
            
            #Filling the unclustered distribution plots
            if colour_clusters == False:
                working[column].plot.kde(ax = ax, label = 'n = '+str(len(working)), color = 'C5')
                median = working[column].median()
                ax.axvline(median, color = 'C5', linestyle = '--')

            #Filling the clustered distribution plots
            else:
                for label in working['cluster'].unique():
                    axis = working.query('cluster == @label')[column]
                    median = axis.median()
                    colour = 'C'+str(label)
                    axis.plot.kde(ax = ax, label = 'n = '+str(len(axis)), color = colour)
                    ax.axvline(median, color = colour, linestyle = '--')
               
            #General subplot organization
            ax.set_yticklabels([])
            ax.set_yticks([])
            ax.set_ylabel(column)
        
        #Collecting legend information from the last subfigure
        line, label = ax.get_legend_handles_labels()
        print(line, label)
        fig.legend = (line, label)
        
        #Setting the x-axis label
        fig.supxlabel(x_label)
        
        #Setting the figure title, it is set as the first axis to make tight_layout()
        #work more simply
        if title == None:
            pass
        else:
            axs[0].set_title(title)
            
        #Cleaning up the layout of the plot
        fig.tight_layout()
        
        #Saving the figure with an appropreate file suffix
        if save_plots == True and filepath != None:
            if colour_clusters == True:
                plt.savefig(filepath+'_distplot_clustered.jpg', dpi = 300)
            else:
                plt.savefig(filepath+'_distplot.jpg', dpi = 300)
        elif save_plots == True and filepath == None:
            print("Please select a valid filepath when calling self.distribution_plots()")   
        else:
            pass             
        
        
    def ternary_plot(self, a, b, c, scale = True):
        """
        This method produces a ternary diagram of the three elements, a, b, and 
        c requested from the arguments. 
        
        The diagram can be scaled by a factor determined by the quotient of the 
        largest mean of the three plotted elements and the element in question. 
        For example if element a had the highest population mean, then the
        scaling factor for b would be μa/μb, and the scaling factor for c would
        be μa/μc. Note that the scaling factor for the population with the highest
        mean will always be 1.
        
        Please note, this method uses the scatter_ternary function from the 
        plotly.express package. Plotly works with html figures, so this figure 
        should plot in your default internet browser. You should not need internet
        access to view the plot. They are somewhat interactive, and can be saved
        permanently from the browser view.

        Parameters
        ----------
        a : str, an element symbol
            One of the three elements of the ternary diagram. Must be present in
            the working dataset
        b : str, an element symbol
            One of the three elements of the ternary diagram. Must be present in
            the working dataset
        c : str, an element symbol
            One of the three elements of the ternary diagram. Must be present in
            the working dataset
        scale : boolean, optional
            Whether to plot scaled data or the raw values. Pass True to scale
            the data, otherwise pass False. The default is True. Note that for
            features with dissparate values (i.e. ppm of 10 to 20 vs ppm of 10 000
            to 20 000), unscaled ternary diagrams may be of limited use as much
            of the data will plot in a small cluster in the corner of the most
            abundant element. In these cases especially passing scale = True is 
            recommended.
        """
        #Selecting the data, and sorting the clusters for proper colouring in plotly
        working = self._working.sort_values('cluster')
        
        #Establishing a ternary frame, maybe a little redundant, but it is to simplify the scaling
        tern_dat = pd.DataFrame()
        
        #Filling the ternary frame, while retaining the element name columns
        for column in working.columns:
            if a in column:
                tern_dat[a] = working[column]
            elif b in column:
                tern_dat[b] = working[column]
            elif c in column:
                tern_dat[c] = working[column]
            else:
                pass


        
        #Producing mean ratios for each corner of the ternary diagram for scaling
        #each ratio is large_mean/small_mean, to get a multiplier that will roughly
        #scale the range of each element to plot helpful ternary diagrams
        mu = (tern_dat.mean().max())/tern_dat.mean()
        #Applying the scaling factor
        tern_dat = tern_dat*mu
        
        #Editing the element column headers to include the scaling factor
        A = a+" * ~"+str(int(mu[0]//1))
        B = b+" * ~"+str(int(mu[1]//1))
        C = c+" * ~"+str(int(mu[2]//1)) 
        tern_dat = tern_dat.rename(columns = {a:A, b:B, c:C})

        #Adding the (unscaled) instance and cluster columns
        tern_dat['cluster'] = working['cluster'].astype('str')
        tern_dat['instance'] = working['instance']

        #Plotting the diagrams
        fig = px.scatter_ternary(data_frame = tern_dat, a = A, b = B, c = C, color = 'cluster', template = 'seaborn', hover_name = 'instance')
        fig.update_traces(marker = {'size': 15})
        plot(fig)

    def save_data(self, data_type, filepath = None):
        """
        Saving class data to a .csv with a clustered column

        Parameters
        ----------
        data_type : string, either 'working' or 'raw'
            Which form of data to save to the .csv, the working dataframe, which
            has much of the data parsed down for plotting and clustering, or the 
            raw data with clusters added. If any other argument is passed an 
            error will be raised
        filepath : string, optional
            The full filepath for the directory in which the data .csv should be 
            saved. If None is entered, this method will use the filepath defined
            by self.filepath or self.folderpath. The default is None.
        """
        #Selecting the dataframe to export
        data = None
        if data_type == 'working':
            data = self._working
        elif data_type == 'raw':
            data = self._raw
        else:
            return print('Please call save_data() with a data_type argument of either "working" or "raw"')
    
        #defining the filepath
        if filepath == None:
            if self._folderpath != None and self._filename != None:
                filepath = self._folderpath+self._filename+'.csv'
            elif self._folderpath != None and self._filename == None:
                filepath = self._folderpath+'_clustered.csv'
            else:
                pass
        else:
            #Checking for and possibly making a directory
            file = filepath.split('/')[-1]
            folder = filepath[:(0-len(file))]
            if not os.path.exists(folder):
                os.makedirs(folder)
        
        #saving the dataframe
        data.to_csv(filepath, index = False)
            
    
    
if __name__ == "__main__":
    test_data = pd.read_csv("D:/ICPMS/Post gis/01A02C_te.csv")
    test_data = test_data.rename(columns = {'X':'x', 'Y':'y', 'Sample':'sample', 'Spot(um)':'spot'})
    test = Kmeans_interrogation()
    test.load(data = test_data, instance_marker = 'Analysis')
    test.remove_nan(nan_threshold=2)
    test.set_filepath("D:/ICPMS/Post gis/Figs/test/foldername/filename")
    test.cluster(clusters = 3, include_elements = ['Na', 'Al', 'K'])
    print(test._subset)
    #test.pair_plot(colour_clusters = True, plot_full = True)
    #test.distribution_plots(title = '01A02C', save_plots = True, x_label = 'ppm abundance', colour_clusters = True)
    #test.ternary_plot(a = 'Li', b = 'Na', c = 'Al')
    #test.save_data(working)