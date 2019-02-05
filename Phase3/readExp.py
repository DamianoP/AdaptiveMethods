import os
import sys
import json
import re
import IPython as ip
import pandas as pd
import numpy as np
import matplotlib as mp
import seaborn as sb
import ck.kernel as ck
import matplotlib.pyplot as plt
import codecs
from matplotlib import cm
from IPython.display import Image, display
verbose         = 1


repoConv        =raw_input("Conv repo:")
repoDirectConv  =raw_input("Directconv repo:")
repoWinog       =""
winogradPresent =raw_input("Winograd present ? 1 / 0 ")
if winogradPresent=="1":
    repoWinog   =raw_input("Winograd repo:")
    winogradPresent=1
else:
    winogradPresent=0

precision       =raw_input("Precision (default or uint8):")
architecture    =raw_input("Architecture (midgard or byfrost):")
fileName        =raw_input("Name for the .arff file in output: ")

if(architecture=="midgard"):
    architecture=0
else:
    architecture=1

if(precision=="default"):
    precision=0
else:
    precision=1





np.set_printoptions(threshold='nan')


firefly_model = 'Rockchip RK3399 Firefly Board (Linux Opensource)\x00'
firefly_name  = 'Firefly RK3399'
firefly_id    = 'firefly'
firefly_gpu   = 'Mali-T860 MP4'
firefly_gpu_mhz = 800

model_to_id = {
    firefly_model : firefly_id,
}
id_to_name = {
    firefly_id : firefly_name,
}
id_to_gpu = {
    firefly_id : firefly_gpu,
}
id_to_gpu_mhz = {
    firefly_id : firefly_gpu_mhz,
}


print ('CK version: %s' % ck.__version__)
print ('IPython version: %s' % ip.__version__)
print ('Pandas version: %s' % pd.__version__)
print ('NumPy version: %s' % np.__version__)
print ('Matplotlib version: %s' % mp.__version__)
print ('Seaborn version: %s' % sb.__version__)

from IPython.display import Image, display
def display_in_full(df):
    pd.options.display.max_columns = len(df.columns)
    pd.options.display.max_rows = len(df.index)
    display(df)
default_colormap = cm.autumn
default_figwidth = 24
default_figdpi = 200
default_fontsize = 16
if mp.__version__[0]=='2': mp.style.use('classic')
mp.rcParams['figure.dpi'] = default_figdpi
mp.rcParams['font.size'] = default_fontsize
mp.rcParams['legend.fontsize'] = 'medium'
mp.rcParams['figure.max_open_warning'] = 200

def get_experimental_results(repo_uoa='local', tags='nntest', profiling=False, skip_synthetic_dataset=True):
    module_uoa = 'experiment'
    r = ck.access({'action':'search', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'tags':tags})
    if r['return']>0:
        print('Error: %s' % r['error'])
        exit(1)
    experiments = r['lst']

    dfs = []
    for experiment in experiments:
        data_uoa = experiment['data_uoa']
        r = ck.access({'action':'list_points', 'repo_uoa':repo_uoa, 'module_uoa':module_uoa, 'data_uoa':data_uoa})
        if r['return']>0:
            print('Error: %s' % r['error'])
            exit(1)
        # Skip experiments if the tags are not in the expected format.
        skip = False
        library = None
        species = None
        # Library tags.
        library_prefix = 'arm-compute-library-'
        library_tags = [ tag[len(library_prefix):] for tag in r['dict']['tags'] if tag.startswith(library_prefix) ]
        if len(library_tags)==1:
            library = library_tags[0]
        else:
            skip = True
        # Species tags.
        species_tags = [ tag for tag in r['dict']['tags'] if tag in ['conv', 'fullyconnected', 'avgpool', 'softmax'] ]
        if len(species_tags)==1:
            species = species_tags[0]
        else:
            skip = True
        # Check if the experiment should be skipped.
        if skip:
            print('[Warning] Skipping experiment with tags:')
            print(r['dict']['tags'])
            continue
        for point in r['points']:
            point_file_path = os.path.join(r['path'], 'ckp-%s.0001.json' % point)
            with open(point_file_path) as point_file:
                point_data_raw = json.load(point_file)
            characteristics_list = point_data_raw['characteristics_list']
            num_repetitions = len(characteristics_list)
            #platform = model_to_id[point_data_raw['features']['platform']['platform']['model']]
            platform="test"
            # Shorten the Git hash to 7 symbols to unify across platforms.
            if platform=='hikey': # hikey_id
                if library_tags[0]=='request-d8f69c13':
                    library = 'opencl-18.03-d8f69c1-request'
                elif library_tags[0]=='opencl-18.05-0acd60ed-request':
                    library = 'opencl-18.05-0acd60e-request'
                else:
                    library = library_tags[0][:-1]
            batch_size = np.int64(point_data_raw['choices']['env'].get('CK_IN_SHAPE_N',-1))
            in_shape_n = np.int64(point_data_raw['choices']['env'].get('CK_IN_SHAPE_N',-1))
            in_shape_c = np.int64(point_data_raw['choices']['env'].get('CK_IN_SHAPE_C',-1))
            in_shape_h = np.int64(point_data_raw['choices']['env'].get('CK_IN_SHAPE_H',-1))
            in_shape_w = np.int64(point_data_raw['choices']['env'].get('CK_IN_SHAPE_W',-1))
            tuner = point_data_raw['choices']['env'].get('CK_LWS_TUNER_TYPE','NONE')
            program = point_data_raw['choices']['data_uoa']
            operator = program[:-len('-armcl-opencl')]
            dataset_uoa = point_data_raw['choices']['dataset_uoa']
            if skip_synthetic_dataset and dataset_uoa.find('synthetic')!=-1: continue
            dataset = point_data_raw['choices']['dataset_file']
            tensor = dataset[len('shape-'):]
            # Obtain column data.
            if profiling: # Obtain kernel time from profiling experiments.
                index = [
                    'platform', 'library', 'operator', 'tensor', 'batch_size', 'kernel', 'repetition_id'
                ]
                data = []
                if point_data_raw['choices'].get('dvdt_prof','')!='':
                    data = [
                        {
                            # features
                            'platform': platform,
                            'library': library,
                            'species': species,
                            # choices
                            'operator' : operator,
                            'tensor' : tensor,
                            'batch_size': batch_size,
                            'in_shape_n': in_shape_n,
                            'in_shape_c': in_shape_c,
                            'in_shape_h': in_shape_h,
                            'in_shape_w': in_shape_w,
                            # statistical repetition
                            'repetition_id': repetition_id,
                            # runtime characteristics
                            'kernel': kernel,
                            'time_us': time_us,
                            'dvdt_prof': characteristics['run'].get('dvdt_prof', {}),
                            'success?': characteristics['run'].get('run_success', 'n/a')
                        }
                        for (repetition_id, characteristics) in zip(range(num_repetitions), characteristics_list)
                        for kernel, time_us in characteristics['run'].get('execution_time_opencl_us',{}).iteritems()
                    ]
                elif point_data_raw['choices'].get('mali_hwc','')!='':
                    data = [
                        {
                            # features
                            'platform': platform,
                            'library': library,
                            'species': species,                            
                            # choices
                            'operator' : operator,
                            'tensor' : tensor,
                            'batch_size': batch_size,
                            'in_shape_n': in_shape_n,
                            'in_shape_c': in_shape_c,
                            'in_shape_h': in_shape_h,
                            'in_shape_w': in_shape_w,
                            # statistical repetition
                            'repetition_id': repetition_id,
                            # runtime characteristics
                            'kernel': 'n/a',
                            'time_us': 0.0,
                            'mali_hwc': characteristics['run'].get('mali_hwc', {}),
                            'success?': characteristics['run'].get('run_success', 'n/a')
                        }
                        for (repetition_id, characteristics) in zip(range(num_repetitions), characteristics_list)
                    ]
                else: # Skip non-profiling experiments.
                    continue
                # Deal with missing data (resulting from failed runs).
                if data==[]:
                    print('[Warning] Missing data for: '+
                          'platform=%s, dataset=%s, library=%s, batch_size=%d' %
                          (platform, dataset, library, batch_size))
                    print(point_file_path)
                    print
                    data = [
                        {
                            # features
                            'platform': platform,
                            'library': library,
                            'species': species,                            
                            # choices
                            'operator' : operator,
                            'tensor' : tensor,
                            'batch_size': batch_size,
                            'in_shape_n': in_shape_n,
                            'in_shape_c': in_shape_c,
                            'in_shape_h': in_shape_h,
                            'in_shape_w': in_shape_w,
                            # statistical repetition
                            'repetition_id': 0,
                            # runtime characteristics
                            'kernel': 'n/a',
                            'time_us': 0.0,
                            'success?': 'n/a'
                        }
                    ]
            else: # Obtain wallclock time from validation experiments.
                if point_data_raw['choices']['dvdt_prof']=='yes':
                    continue # Skip profiling experiments.
                data = [
                    {
                        # features
                        'platform': platform,
                        'library': library,
                        'species': species,                        
                        # choices
                        'tuner' : tuner,                        
                        'operator' : operator,
                        'tensor' : tensor,
                        'batch_size': batch_size,
                        'in_shape_n': in_shape_n,
                        'in_shape_c': in_shape_c,
                        'in_shape_h': in_shape_h,
                        'in_shape_w': in_shape_w,
                        # statistical repetition
                        'repetition_id': repetition_id,
                        # runtime characteristics
                        'time_us': 1e6*characteristics['run'].get('execution_time_kernel_1',0.0),
                        'success?': characteristics['run'].get('run_success', 'n/a')
                    }
                    for (repetition_id, characteristics) in zip(range(num_repetitions), characteristics_list)
                ]
                index = [
                    'platform', 'library', 'operator', 'tensor', 'batch_size', 'tuner', 'repetition_id'
                ]
            # Construct a DataFrame.
            df = pd.DataFrame(data)
            # Calculate GFLOPS for conv and fullyconnected species. NB: 2 operations per element (multiply and accumulate).
            if species=='conv':
                flops = 2 * df['tensor'] \
                    .apply(lambda tensor : np.float64(tensor.split('-'))) \
                    .apply(lambda (in_C, H, W, K, out_C, stride, pad) : in_C*out_C*(W/stride)*(H/stride)*K*K) \
                    .values
            elif species=='fullyconnected':
                flops = 2 * df['tensor'] \
                    .apply(lambda tensor : np.float64(tensor.split('-'))) \
                    .apply(lambda (in_C, in_H, in_W, out_C, out_H, out_W) : (1, in_C*in_H*in_W, out_C*out_H*out_W)) \
                    .apply(lambda (M, K, N): M*K*N) \
                    .values
            else:
                flops = 0
            Gflops = 1e-9 * flops          # 1 Gflops == 1e+9 flops.
            time_s = 1e-6 * df['time_us']  # 1 second == 1e+6 microseconds.
            df['GFLOPS'] = Gflops / time_s # GFLOPS == Gflops per second.
            # Set index.
            df = df.set_index(index)
            # Append to the list of similarly constructed DataFrames.
            dfs.append(df) 
    if dfs:
        # Concatenate all thus constructed DataFrames (i.e. stack on top of each other).
        result = pd.concat(dfs)
        result = result.sort_index(level=result.index.names)
    else:
        # Construct a dummy DataFrame which success status can be safely checked.
        result = pd.DataFrame(columns=['success?', 'time_us', 'GFLOPS'])
    return result

# groupby_level: 'platform', 'operator', 'library' or 'kernel' (with dvdt-prof).
# Typically, when creating df_raw, we consider one of the following scenarios:
# - different platforms (e.g. hikey and mate), same library (e.g. v18.05), same operator (e.g. fullyconnected)
# - different operators (e.g. conv, directconv, winogradconv), same library (e.g. v18.05), same platform (e.g. mate)
# - different libraries (e.g. v18.03, v18.05), same operator (e.g. directconv), same platform (e.g. hikey)
def plot_violin(df_raw, groupby_level='operator', species=None, metric='time_us',
                title=None, figsize=None, fontscale=1.75, legend_loc='upper right',
                platform_id=firefly_id, gpu=firefly_gpu, gpu_mhz=firefly_gpu_mhz,
                xmin=None, xmax=None, xstep=None):
    # Get species.
    if not species:
        if(df_raw['species']=='fullyconnected').all():
            species = 'fullyconnected'
        elif(df_raw['species']=='conv').all():
            species = 'conv'
        elif(df_raw['species']=='avgpool').all():
            species = 'avgpool'
        elif(df_raw['species']=='softmax').all():
            species = 'softmax'
        else:
            print('Warning: unknown or mixed species')
    # Set depending on groupby_level. Drop batch_size (batch_size==1). TODO: support batch_size!=1.
    if groupby_level=='platform':
        tuples = [ (p,'%s;%s;%s'%(o,l,u),t,r) for (p,l,o,t,b,u,r) in df_raw.index.values ]
        names = [ 'platform', 'operator;library;tuner', 'tensor', 'repetition_id' ]
        hue_levels = 'operator;library;tuner'
        palette = 'summer'
    elif groupby_level=='operator':
        tuples = [ (o,'%s;%s;%s'%(p,l,u),t,r) for (p,l,o,t,b,u,r) in df_raw.index.values ]
        names = [ 'operator', 'platform;library;tuner', 'tensor', 'repetition_id' ]
        hue_levels = 'platform;library;tuner'
        palette = 'autumn'
    elif groupby_level=='library':
        tuples = [ (l,'%s;%s;%s'%(o,p,u),t,r) for (p,l,o,t,b,u,r) in df_raw.index.values ]
        names = [ 'library', 'operator;platform;tuner', 'tensor', 'repetition_id' ]
        hue_levels = 'operator;platform;tuner'
        palette = 'spring'
    elif groupby_level=='kernel': # no tuner
        tuples = [ (k,'%s;%s;%s'%(o,p,l),t,r) for (p,l,o,t,b,k,r) in df_raw.index.values ]
        names = [ 'kernel', 'operator;platform;library', 'tensor', 'repetition_id' ]
        hue_levels = 'operator;platform;library'
        palette = 'winter'
    else:
        print('Error: unsupported groupby_level=%s' % groupby_level)
        exit(1)
    # Create a new DataFrame using the index settings above.
    df_violin = pd.DataFrame(data=df_raw[metric].values, columns=[metric],
                             index=pd.MultiIndex.from_tuples(tuples=tuples, names=names)).sort_index()
    # Set style.
    sb.set_style('whitegrid')
    sb.set_palette(palette)
    fontsize = default_fontsize*fontscale
    if metric=='time_us':
        xlabel = 'Operator execution time (microseconds)'
    elif metric=='GFLOPS':
        xlabel = 'Operator GFLOPS (billion single-precision floating-point operations per second)'
    ylabel = 'Tensor shape'
    if species=='fullyconnected':
        ylabel += ' (in_C, in_H, in_W, out_C, out_H, out_W)'
    elif species=='conv':
        ylabel += ' (in_C, H, W, K, out_C, stride, pad)'
    elif species=='avgpool':
        ylabel += ' (C, H, W, K, stride)'
    elif species=='softmax':
        ylabel += ' (C, H, W)'
    if not figsize:
        num_groupby_values = len(df_raw.index.get_level_values(level=groupby_level).unique())
        # TODO: num_kernels
        num_platforms = len(df_raw.index.get_level_values(level='platform').unique())
        num_libraries = len(df_raw.index.get_level_values(level='library').unique())
        num_operators = len(df_raw.index.get_level_values(level='operator').unique())
        num_tensors = len(df_raw.index.get_level_values(level='tensor').unique())
        num_tuners = len(df_raw.index.get_level_values(level='tuner').unique())
        figheight = num_platforms * num_libraries * num_operators * num_tensors * num_tuners / num_groupby_values
        figsize = (default_figwidth, figheight)
    # For each unique groupby value.
    groupby_values = df_violin.index.get_level_values(level=groupby_level).unique()
    for groupby_value in groupby_values:
        df_violin_loc = df_violin.loc[groupby_value].reset_index()
        fig = plt.figure(figsize=figsize, dpi=default_figdpi)
        ax = fig.gca()
        sb.violinplot(ax=ax, data=df_violin_loc, x=metric, y='tensor', inner='point', split=False, saturation=0.8,
                      hue=hue_levels, fontscale=fontscale)
        # Title.
        if title:
            groupby_title = '%s: %s=%s' % (title, groupby_level, groupby_value)
        else:
            if groupby_level=='platform':
                groupby_title = '%s (GPU: %s @ %d MHz)' % (id_to_name[platform_id], gpu, gpu_mhz)
            else:
                groupby_title = '%s=%s' % (groupby_level, groupby_value)
        ax.set_title(groupby_title, fontsize=fontsize)
        # X axis.
        if not xstep: xstep = 1000
        if not xmin: xmin = np.int64(df_violin_loc[metric].min())
        if not xmax: xmax = np.int64(df_violin_loc[metric].max()) // xstep * xstep + xstep
        ax.set_xlim([xmin, xmax])
        ax.set_xticks(range(0, xmax+1, xstep))
        ax.set_xlabel(xlabel, fontsize=fontsize)
        # Y axis.
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)
        # Vertical lines between groups of violins.
        for y in ax.get_yticks():
            ax.hlines(y=y+0.5, xmin=0, xmax=xmax, linestyles='dotted', colors='black')
        # Legend location.
        ax.legend(loc=legend_loc);

def plot_speedup(df, baseline_levels='platform', baseline_values=firefly_id,
                 title=None, xmax=None, figsize=None, fontsize=default_fontsize,
                 legend_loc='lower right'):
    if not title: title = '%s=%s' % (str(baseline_levels), str(baseline_values))
    if not figsize:
        num_rows = len(df.index)
        num_tensors = len(df.index.get_level_values(level='tensor').unique())
        fig_height = 2 * num_rows / num_tensors
        figsize=[default_figwidth, fig_height]
    df_ = df.unstack(baseline_levels)
    df_speedup = 1. / df_.divide(df_[baseline_values], axis=0).stack(baseline_levels)
    # Plot the speedups.
    axes = pd.DataFrame(data=df_speedup, columns=['speedup (x)']) \
        .reorder_levels(['tensor', 'platform', 'library', 'operator', 'batch_size', 'tuner']) \
        .groupby(level=['tensor']) \
        .plot(kind='barh', title=title, stacked=False, width=0.8, grid=True, legend=True,
              xlim=[0,xmax], figsize=figsize, fontsize=default_fontsize, colormap=cm.autumn)
    # Annotate the bars with the speedups rounded to one digit after the decimal point.
    for ax in axes:
        ax.set_title(title, fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.legend(loc=legend_loc)
        for patch in ax.patches:
            text = '{0:.1f}x'.format(patch.get_width())
            ax.annotate(text, (patch.get_width()*1.01, patch.get_y()*1.01), fontsize=64)



def is_present(arr,item):
    for a in range(0,len(arr)):
        if arr[a]==item:
            return True
    return False

def writeStringToFile(file,arr):
    splitTensor=arr[2].split("-")
    newRow=""
    for i in range(0,len(splitTensor)):
        newRow=newRow+splitTensor[i]+","
    if str(arr[3])=="NONE":
        tunerValue="0" # "false"
    else :
        tunerValue="1" # "true"
    #newRow=newRow+str(arr[9])+","+str(tunerValue)+","+str(arr[1])+","+str(precision)+","+str(architecture)+"\n")
    newRow=newRow+str(tunerValue)+","+str(precision)+","+str(architecture)+","+str(arr[1])+"\n"
    # no computation time in the file
    file.write(newRow)
    print(newRow)





print("reading experiments: convolution")
df_conv = get_experimental_results(repo_uoa=repoConv, tags='conv', profiling=False)
df_conv = df_conv[df_conv['success?']=='yes']
#elements_conv = df_conv['time_us'].groupby(level=df_conv.index.names[:-1]).describe()
elements_conv = df_conv['time_us'].groupby(["platform","operator","tensor","tuner"]).describe()
rawData_conv = elements_conv.to_records() #all data collected and usable with rawData[i]



print("reading experiments: direct_convolution")
df_direct = get_experimental_results(repo_uoa=repoDirectConv, tags='conv', profiling=False)
df_direct = df_direct[df_direct['success?']=='yes']
#elements_direct = df_direct['time_us'].groupby(level=df_direct.index.names[:-1]).describe()
elements_direct =df_direct['time_us'].groupby(["platform","operator","tensor","tuner"]).describe()
rawData_direct =elements_direct.to_records() #all data collected and usable with rawData[i]



if winogradPresent==1:
    print("reading experiments: winograd")
    df_winograd = get_experimental_results(repo_uoa=repoWinog, tags='conv', profiling=False)
    df_winograd = df_winograd[df_winograd['success?']=='yes']
    #elements_winograd = df_winograd['time_us'].groupby(level=df_winograd.index.names[:-1]).describe()
    elements_winograd = df_winograd['time_us'].groupby(["platform","operator","tensor","tuner"]).describe()
    rawData_winograd = elements_winograd.to_records() #all data collected and usable with rawData[i]
else:
    rawData_winograd = []

if verbose==1:
    print(rawData_conv)
    print(rawData_direct)
    print(rawData_winograd)

rawData_conv_lenght     =   len(rawData_conv)-1
rawData_direct_lenght   =   len(rawData_direct)-1
rawData_winograd_lenght =   len(rawData_winograd)-1

results=[]

workingMemory=[]

workingMemory.append(rawData_conv)
workingMemory.append(rawData_direct)
workingMemory.append(rawData_winograd)

print("Creating file..."+fileName+".arff")
file=open("results/"+fileName+".arff","w")
file.write("@RELATION convolution\n")
file.write("@ATTRIBUTE tensor_1 INTEGER\n")
file.write("@ATTRIBUTE tensor_2 INTEGER\n")
file.write("@ATTRIBUTE tensor_3 INTEGER\n")
file.write("@ATTRIBUTE tensor_4 INTEGER\n")
file.write("@ATTRIBUTE tensor_5 INTEGER\n")
file.write("@ATTRIBUTE tensor_6 INTEGER\n")
file.write("@ATTRIBUTE tensor_7 INTEGER\n")
#file.write("@ATTRIBUTE computationTime REAL\n")
file.write("@ATTRIBUTE tuner INTEGER\n")
file.write("@ATTRIBUTE precision INTEGER\n")
file.write("@attribute architecture INTEGER\n")
file.write("@ATTRIBUTE class {conv,directconv,winogradconv}\n")
file.write("\n")
file.write("\n")
file.write("\n")
file.write("@DATA\n")

print("Creating file..."+fileName+"_ranking.txt")
file2=open("results/"+fileName+"_ranking.txt","w")
#for i in range(0,rawData_conv_lenght):
#    if is_present(results,rawData_conv[i][3])==False #tensor (shape)
rank1=""
rank2=""
rank3=""
speed1=""
speed2=""
speed3=""

def insert(listRank,rank):
    newList=[]
    done=0
    for i in range(0,len(listRank)):
        if(float(listRank[i][1][0])<float(rank[1][0])):
            newList.append(listRank[i])
        elif(float(listRank[i][1][0])>=float(rank[1][0]) and done==0):
            newList.append(rank)
            newList.append(listRank[i])
            done=1
        else:
            newList.append(listRank[i])
    if (done==0):
        newList.append(rank)
    #print newList
    return newList


for i in range(0,len(workingMemory)): #scan the 3 arrays
    rawData=workingMemory[i]
    for j in range(0,len(rawData)): # scan the array
        if is_present(results,rawData[j][2])==False:
            tensor=str(rawData[j][2])
            print("Reading: "+tensor)
            results.append(tensor)
            #starting the comparation
            best=rawData[j] # temp best
            rank1="[['"+str(rawData[j][1])+"'],['"+str(rawData[j][9]) +"']]"
            speed1=rawData[j][9]
            for z in range(0,len(workingMemory)): #scan for comparation
                if i==z:
                    continue
                else:
                    rawDataComp=workingMemory[z]
                    for h in range(0,len(rawDataComp)):     #scan the array
                        if  rawData[j][2]==rawDataComp[h][2] :
                            if rawDataComp[h][9]<best[9] and rawData[j][3]==rawDataComp[h][3]:
                                best=rawDataComp[h]
                            if rank2!="":
                                rank3="[['"+str(rawDataComp[h][1])+"'],['"+str(rawDataComp[h][9]) +"']]"
                                speed3=rawDataComp[h][9]
                            if rank2=="":
                                rank2="[['"+str(rawDataComp[h][1])+"'],['"+str(rawDataComp[h][9]) +"']]"
                                speed2=rawDataComp[h][9]
                            h=len(rawDataComp)+1
            tempSpeed   = ""
            tempRank    = ""
            newFileRow=[]
            newFileRow.append(eval("['"+rawData[j][2]+"']\n"))
            listRank=[]
            if(rank1!=""):
                rank1=eval(rank1)
                listRank.append(rank1)

            if(rank2!=""):
                rank2=eval(rank2)
                listRank = insert(listRank, rank2)
            
            if(rank3!=""):
                rank3=eval(rank3)
                listRank = insert(listRank, rank3)

            #print listRank
            newFileRow.append(listRank)
            writeStringToFile(file,best)
            file2.write(str(newFileRow))
            file2.write("\n")
            rank1=""
            rank2=""
            rank3=""
            speed1=""
            speed2=""
            speed3=""
file.close()
file2.close()
print("Done !")










