import string
import sys,os

filter={}
results={}
all_data_name=[]
for filename in ['INEX_LD_v2.txt','ListSearch_v2.txt','QALD2_v2.txt','SemSearch_ES_v2.txt']:
    with open('E:\\Entity_Retrieval\\query\\simple_cluster\\%s'%(filename),'r',encoding='utf-8') as src:
         data_name=filename.replace('.txt','')
         results[data_name]=[]
         all_data_name.append(data_name)
         for line in src:
             item=line.strip().split('\t')
             qid=item[0]
             filter[qid]=data_name

run_name='pylucene_all_mp.runs'
with open(run_name,'r',encoding='utf-8') as dest:
     for line in dest:
         qid=line.strip().split('\t')[0]
         data_name=filter[qid]
         results[data_name].append(line)

for data_name in all_data_name:
    with open('%s.runs'%(data_name),'w',encoding='utf-8') as dest:
         dest.writelines(results[data_name])
         
for dataset in ['INEX_LD','ListSearch','QALD2','SemSearch_ES']:
    
    runs_name=dataset+'_v2.runs'
    runs={} # runs[qid]=line
    with open(runs_name,'r',encoding='utf-8') as dest:
         for line in dest:
             qid=line.split('\t')[0]
             if qid not in runs:
                runs[qid]=[]
             runs[qid].append(line)
    
    os.mkdir(dataset)
    
    evals={} # eval['ndcg_cut_10'] ..
    for i in range(5):
        test_name=os.path.join('E:\\Entity_Retrieval\\five-fold',dataset,str(i),'testing.txt')
        os.mkdir(os.path.join(dataset,str(i)))
        fold_runs_name=os.path.join(dataset,str(i),dataset+'_v2_%d.runs'%(i))
        dest=open(fold_runs_name,'w',encoding='utf-8')
        with open(test_name,'r',encoding='utf-8') as src:
             for qid in src:
                 if qid.strip() not in runs:
                    continue
                 dest.writelines(runs[qid.strip()])
        dest.close()
        
        eval_file_name=os.path.join(dataset,str(i),dataset+'_v2_result_%d.txt'%(i))
        cmd='trec_eval -m map_cut.100 -m ndcg_cut.10,100 %s %s > %s'%('E:\\qrels-v2.txt',fold_runs_name,eval_file_name)
        os.system(cmd)
        
        with open(eval_file_name,'r') as src:
             for line in src:
                 list_item=line.strip().split('\t')
                 metric=list_item[0].strip()
                 value=float(list_item[2].strip())
                 if metric not in evals:
                    evals[metric]=0.0
                 evals[metric]+=value
                 
    with open(dataset+'_v2_five_fold_results.txt','w',encoding='utf-8') as dest:
         for metric in evals:
             dest.write('%s\t%f\n'%(metric,evals[metric]/5))