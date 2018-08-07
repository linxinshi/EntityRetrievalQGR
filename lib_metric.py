import math, gensim
from config import *
from lib_process import findOneDBEntry
from list_term_object import List_Term_Object

def wordSim_tswe(term1,term2,cat,mongoObj):
    conn=mongoObj.conn_tswe
    item1=conn.find_one({'category':cat,'word':term1})
    if item1 is None:
       return 0.0
    item2=conn.find_one({'category':cat,'word':term2})
    if item2 is None:
       return 0.0
    
    v1=item1['vector']
    v2=item2['vector']
    sim=0.0
    for i in range(30):
        sim+=(v1[i]*v2[i])
    print ('%s,%s,%f'%(term1,term2,sim))
    if sim>=0.85:
       return sim
    else:
       return 0.0

def wordSim(term1,term2,w2vmodel,bound):
    if term1==term2:
       return 1.0
    if term1 not in w2vmodel or term2 not in w2vmodel:
       return 0.0
    sim=w2vmodel.similarity(term1,term2)
    #print ('%s,%s,%f'%(term1,term2,sim))
    '''
    try:
       sim=w2vmodel.similarity(term1,term2)
    except:
       sim=0.0
    '''
    if sim>=bound:
       return sim
       #return 1
    else:
       return 0.0

def get_dirichlet_prob(tf_t_d, len_d, tf_t_C, len_C, mu):
    """
    Computes Dirichlet-smoothed probability
    P(t|theta_d) = [tf(t, d) + mu P(t|C)] / [|d| + mu]

    :param tf_t_d: tf(t,d)
    :param len_d: |d|
    :param tf_t_C: tf(t,C)
    :param len_C: |C| = \sum_{d \in C} |d|
    :param mu: \mu
    :return:
    """
    if mu == 0:  # i.e. field does not have any content in the collection
       return 0
    else:
       p_t_C = tf_t_C / len_C if len_C > 0.0 else 0.0
       return (tf_t_d + mu * p_t_C) / (len_d + mu)

       
def mlm_sas2(queryObj,entityObj,structure,lucene_handler):
    if len(entityObj.categories)==0:
       return NEGATIVE_INFINITY
    D=structure.cat_dag
    lucene_cat=lucene_handler['category_corpus']
    lucene_doc=lucene_handler['first_pass']
    
    len_d_f=entityObj.lengths
    max_score=NEGATIVE_INFINITY
    len_C_f={}
    mlm_weights={}
    mu={}
    
    for field in LIST_F:
        len_C_f[field]=lucene_doc.get_coll_length(field)
        mu[field]=lucene_doc.get_avg_len(field)
        mlm_weights[field]=1.0/len(LIST_F)
        
    if MODEL_NAME=='mlm-tc':
       mlm_weights={'stemmed_names':0.2,'stemmed_catchall':0.8} if USED_QUERY_VERSION=='stemmed_raw_query' else {'names':0.2,'attributes':0.8} 

    def smooth_path(cat,path_len):
        nonlocal D,cnt_path
        nonlocal lucene_cat
        nonlocal cf_g_f,len_c_g_f,n_c_g
        
        visited.add(cat)
        if cnt_path>TOP_PATH_NUM_PER_CAT:
           return
        # the following is end condition
        if path_len==LIMIT_SAS_PATH_LENGTH or len(D[cat])==0:
           return
           
        # maintain useful temporary variables
        d,docID=lucene_cat.findDoc(cat,'category',True)
        cnt_doc_corpus=0
        
        if d is not None:
           # maintain
           cnt_doc_corpus=int(d['num_articles'])
           n_c_g+=cnt_doc_corpus
           for f in LIST_F:
               # get category corpus
               term_freq=lucene_cat.get_term_freq(docID,f,True)
               len_c=sum(term_freq.values())
               len_c_g_f[f]+=len_c
               
               for j in range(queryObj.contents_obj.length):
                   term=queryObj.contents_obj.term[j]
                   tf_c=term_freq.get(term,0.0)
                   cf_g_f[f][j]+=tf_c                 
                                     
        cnt=0
        for child in iter(D[cat]):
            cnt+=1
            if cnt>TOP_CATEGORY_NUM:
               break
            if child in visited:
               continue
            if child in D:
               smooth_path(child,path_len+1)

    # end of function smooth_path
    visited=set()
    for cat in entityObj.categories[:TOP_CATEGORY_NUM]:
        if cat not in D:
           continue
        n_c_g=0   
        len_c_g_f={}        
        cf_g_f={}  # collection frequency for each term in each field
        for f in LIST_F:
            len_c_g_f[f]=0
            cf_g_f[f]=[0 for i in range(queryObj.contents_obj.length)]  
        cnt_path=0
        visited.clear()
        
        smooth_path(cat,0)
        
        mu_g_f={}
        for f in LIST_F:
            mu_g_f[f]=len_c_g_f[f]/n_c_g if n_c_g>0 else 0.0
        if n_c_g==0:
           continue
        #cof=0.003
        cof=0.05
        #cof=0
        score=0.0
        for j in range(queryObj.contents_obj.length):
            term=queryObj.contents_obj.term[j]
            ptd=0.0
            for f in LIST_F:
                tf_d_f=entityObj.term_freqs[f].get(term,0.0)
                cf_f = lucene_doc.get_coll_termfreq(term, f)
                ptc_doc=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                ptc_g_f=cf_g_f[f][j]/len_c_g_f[f] if len_c_g_f[f]>0 else 0.0
                if f in NO_SMOOTHING_LIST:
                   ptd_f=(tf_d_f+mu[f]*ptc_doc)/(len_d_f[f]+mu[f]) if (len_d_f[f]+mu[f])>0 else 0.0                                   
                else:
                   ptd_f=(tf_d_f+mu[f]*ptc_doc+cof*mu_g_f[f]*ptc_g_f)/(len_d_f[f]+mu[f]+cof*mu_g_f[f]) if (len_d_f[f]+mu[f]+cof*mu_g_f[f])>0 else 0.0             
                #ptd_f=(tf_d_f+mu[f]*ptc_doc)/(len_d_f[f]+mu[f]) if (len_d_f[f]+mu[f])>0 else 0.0
                ptd+=mlm_weights[f]*ptd_f
            if ptd>0:
               score+=math.log(ptd)
        max_score=max(score,max_score)
           
    return max_score    

def fsdm_sas2(queryObj,entityObj,structure,lucene_handler):
    if len(entityObj.categories)==0:
       return NEGATIVE_INFINITY
    D=structure.cat_dag
    lucene_cat=lucene_handler['category_corpus']
    lucene_doc=lucene_handler['first_pass']
    
    len_d_f=entityObj.lengths
    max_score=NEGATIVE_INFINITY
    len_C_f={}
    mlm_weights={}
    mu={}
    
    for field in LIST_F:
        len_C_f[field]=lucene_doc.get_coll_length(field)
        mu[field]=lucene_doc.get_avg_len(field)
        mlm_weights[field]=1.0/len(LIST_F)
        
    if MODEL_NAME=='mlm-tc':
       mlm_weights={'stemmed_names':0.2,'stemmed_catchall':0.8} if USED_QUERY_VERSION=='stemmed_raw_query' else {'names':0.2,'attributes':0.8} 

       
    w={} # (term,ordered)={field:field_weight}
    for t in queryObj.contents_obj.term:
        w[(t,True)]=get_mapping_prob(t,lucene_doc)
    for bigram_pair in queryObj.bigrams:
        bigram=bigram_pair[0]+' '+bigram_pair[1]
        w[(bigram,True)]=get_mapping_prob(bigram,lucene_doc,ordered=True,slop=0)
    for bigram_pair in queryObj.bigrams:
        bigram=bigram_pair[0]+' '+bigram_pair[1]
        w[(bigram,False)]=get_mapping_prob(bigram,lucene_doc,ordered=False,slop=6)

    def smooth_path(cat,path_len):
        nonlocal D,cnt_path
        nonlocal lucene_cat
        nonlocal cf_g_f,len_c_g_f,n_c_g
        
        visited.add(cat)
        if cnt_path>TOP_PATH_NUM_PER_CAT:
           return
        # the following is end condition
        if path_len==LIMIT_SAS_PATH_LENGTH or len(D[cat])==0:
           return
           
        # maintain useful temporary variables
        d,docID=lucene_cat.findDoc(cat,'category',True)
        cnt_doc_corpus=0
        
        if d is not None:
           # maintain
           cnt_doc_corpus=int(d['num_articles'])
           n_c_g+=cnt_doc_corpus
           for f in LIST_F:
               # get category corpus
               term_freq=lucene_cat.get_term_freq(docID,f,True)
               len_c=sum(term_freq.values())
               len_c_g_f[f]+=len_c
               
               for j in range(queryObj.contents_obj.length):
                   term=queryObj.contents_obj.term[j]
                   tf_c=term_freq.get(term,0.0)
                   cf_g_f[f][j]+=tf_c                 
                                     
        cnt=0
        for child in iter(D[cat]):
            cnt+=1
            if cnt>TOP_CATEGORY_NUM:
               break
            if child in visited:
               continue
            if child in D:
               smooth_path(child,path_len+1)

    # end of function smooth_path
    visited=set()
    for cat in entityObj.categories[:TOP_CATEGORY_NUM]:
        if cat not in D:
           continue
        n_c_g=0   
        len_c_g_f={}        
        cf_g_f={}  # collection frequency for each term in each field
        for f in LIST_F:
            len_c_g_f[f]=0
            cf_g_f[f]=[0 for i in range(queryObj.contents_obj.length)]  
        cnt_path=0
        visited.clear()
        
        smooth_path(cat,0)
        
        mu_g_f={}
        for f in LIST_F:
            mu_g_f[f]=len_c_g_f[f]/n_c_g if n_c_g>0 else 0.0
        if n_c_g==0:
           continue
           
        cof=0.05
        ft_p=0.0
        for j in range(queryObj.contents_obj.length):
            term=queryObj.contents_obj.term[j]
            ptd=0.0
            for f in w[(term,True)]:
                tf_d_f=entityObj.term_freqs[f].get(term,0.0)
                cf_f = lucene_doc.get_coll_termfreq(term, f)
                ptc_doc=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                ptc_g_f=cf_g_f[f][j]/len_c_g_f[f] if len_c_g_f[f]>0 else 0.0
                if f in NO_SMOOTHING_LIST:
                   ptd_f=(tf_d_f+mu[f]*ptc_doc)/(len_d_f[f]+mu[f]) if (len_d_f[f]+mu[f])>0 else 0.0                                   
                else:
                   ptd_f=(tf_d_f+mu[f]*ptc_doc+cof*mu_g_f[f]*ptc_g_f)/(len_d_f[f]+mu[f]+cof*mu_g_f[f]) if (len_d_f[f]+mu[f]+cof*mu_g_f[f])>0 else 0.0             
                ptd+=w[(term,True)][f]*ptd_f
            if ptd>0:
               ft_p+=math.log(ptd)
               
        # for ordered bigrams
        fo_p=0.0
        if LAMBDA_O>0:
           for j in range(len(queryObj.bigrams)):
               bigram=queryObj.bigrams[j][0]+' '+queryObj.bigrams[j][1]
               ptd=0.0
               for f in w[(bigram,True)]:
                   tf_d_f,cf_f=lucene_doc.get_coll_bigram_freq(bigram,f,True,0,entityObj.title)
                   ptc_f=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                   Dt=mu[f]*ptc_f
                   Nt=mu[f]
                   ptd_f=(tf_d_f+Dt)/(len_d_f[f]+Nt) if len_d_f[f]+Nt>0 else 0.0
                   ptd+=w[(bigram,True)][f]*ptd_f
               if ptd>0:
                  fo_p+=math.log(ptd)*w[(bigram,True)][f]           
        # for unordered bigrams
        fu_p=0.0
        if LAMBDA_U>0:
           for j in range(len(queryObj.bigrams)):
               bigram=queryObj.bigrams[j][0]+' '+queryObj.bigrams[j][1]
               ptd=0.0
               for f in w[(bigram,False)]:
                   tf_d_f,cf_f=lucene_doc.get_coll_bigram_freq(bigram,f,False,6,entityObj.title)
                   ptc_f=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                   Dt=mu[f]*ptc_f
                   Nt=mu[f]
                   ptd_f=(tf_d_f+Dt)/(len_d_f[f]+Nt) if len_d_f[f]+Nt>0 else 0.0
                   ptd+=w[(bigram,False)][f]*ptd_f
               if ptd>0:
                  fu_p+=math.log(ptd)*w[(bigram,False)][f]   
                  
        score_p=LAMBDA_T*ft_p+LAMBDA_O*fo_p+LAMBDA_U*fu_p
        max_score=max(score_p,max_score)
           
    return max_score   
    
def elrSim(queryObj,entityObj,lucene_obj,mongoObj):
    # only catchall field
    field='catchall'
    df_f=lucene_obj.get_doc_count(field)
    score=NEGATIVE_INFINITY
    
    for entity in queryObj.query_entities:
        se=queryObj.query_entities[entity]
        
        d,docID=lucene_obj.findDoc(entity,'title',True)
        if d is None or docID is None:
           continue
        #print ('cur:'+entity)
        
        title=entity.lower()
        term_freq=lucene_obj.get_term_freq(docID,field,False)
        #print (term_freq)
        tf=1 if title in term_freq else 0
        df_e_f=lucene_obj.get_doc_freq(title, field)
        
        #print (df_e_f,df_f)
        cur_score=NEGATIVE_INFINITY
        sim=0.9*tf+0.1*(df_e_f/df_f)
        #print (se)
        #print (sim)
        
        if sim>0:
           cur_score=se*math.log(sim)
        if cur_score>score:
           score=cur_score
           
    if score==NEGATIVE_INFINITY:
       return 0.0
    else:
       #print ('score=%f'%(score))
       return score
    
def bm25fSim(lt_obj1,entityObj,lucene_obj):
    len_C_f={}
    mu={}
    for f in LIST_F:
        len_C_f[f]=lucene_obj.get_coll_length(f)
        mu[f]=lucene_obj.get_avg_len(f)
        
    N=lucene_obj.get_doc_count('stemmed_catchall')    
    k1=2.44
    b=0.297
    boost=1
    
    totalSim=0.0
    for i in range(lt_obj1.length):
        term=lt_obj1.term[i]
        localSim=0.0
        df_t=lucene_obj.get_doc_freq(term,'stemmed_catchall')
        idf_t=math.log10((N-df_t+0.5)/(df_t+0.5))
        weight_t_d=0.0
        
        for f in LIST_F:
            len_d_f = entityObj.lengths[f]
            tf_t_d_f = entityObj.term_freqs[f].get(term,0)
            tf_t_C_f = lucene_obj.get_coll_termfreq(term, f)
            
            # compute f(p(t1|De),p(t2|De)...) 
            weight_t_d+=((tf_t_d_f*boost)/(1-b+b*(len_d_f/mu[f])))
        totalSim+=(idf_t*(weight_t_d/(k1+weight_t_d)))    
    return totalSim
    
def lmSim(lt_obj1,entityObj,field,w2vmodel,lucene_obj,mongoObj=None):
    # subquery x et[0..n-1] 
    totalSim=0.0
    term_freq=entityObj.term_freq
    len_C_f = lucene_obj.get_coll_length(field)
    mu=lucene_obj.get_avg_len(field)
    cnt=0
    
    # iterate each t in term_freq and compare similarity
    for i in range(lt_obj1.length):
        qt=lt_obj1.term[i]
        localSim=0.0
        # compute p(t|De)
        if WORD_EMBEDDING_TYPE=='NONE':
           if qt in term_freq:
              localSim=term_freq[qt]
        else:
           # use embedding
           for et in term_freq:
               localSim+=(wordSim(qt,et,w2vmodel,BOUND_SIM)*term_freq[et])
           
        if localSim>0.0:
           cnt+=1
           
        len_d_f = entityObj.length
        tf_t_d_f = localSim
        tf_t_C_f = lucene_obj.get_coll_termfreq(qt, field)
        
        p_t_d=get_dirichlet_prob(tf_t_d_f, float(len_d_f), float(tf_t_C_f), float(len_C_f), mu)
        # compute f(p(t1|De),p(t2|De)...) 
        if p_t_d>0.0:
           totalSim+=math.log(p_t_d)
    return totalSim
    
    
def mlmSim(lt_obj1,entityObj,lucene_obj,mongoObj=None):
    # need every field representation instead of single lt_obj for entity
    # subquery x et[0..n-1] 
    
    len_C_f={}
    mu={}
    mlm_weights={}
    for f in LIST_F:
        len_C_f[f]=lucene_obj.get_coll_length(f)
        mu[f]=lucene_obj.get_avg_len(f)
        mlm_weights[f]=1.0/len(LIST_F)
        
    if MODEL_NAME=='mlm-tc':
       mlm_weights={'stemmed_names':0.2,'stemmed_catchall':0.8} if USED_QUERY_VERSION=='stemmed_raw_query' else {'names':0.2,'catchall':0.8} 
        
    totalSim=0.0
    for i in range(lt_obj1.length):
        qt=lt_obj1.term[i]
        localSim=0.0
        # compute p(t|Df)
        for f in LIST_F:
            len_d_f = entityObj.lengths[f]
            tf_t_d_f = entityObj.term_freqs[f].get(qt,0)
            tf_t_C_f = lucene_obj.get_coll_termfreq(qt, f)
            
            tempSim=get_dirichlet_prob(tf_t_d_f, len_d_f, tf_t_C_f, len_C_f[f], mu[f])
            # compute f(p(t1|De),p(t2|De)...) 
            localSim+=mlm_weights[f]*tempSim
            
        if localSim>0.0:
           totalSim+=math.log(localSim)
    return totalSim

def prmsSim(lt_obj1,entityObj,lucene_obj):
    # need every field representation instead of single lt_obj for entity
    # subquery x et[0..n-1] 
    
    len_C_f={}
    mu={}
    for f in LIST_F:
        len_C_f[f]=lucene_obj.get_coll_length(f)
        mu[f]=lucene_obj.get_avg_len(f)
               
    totalSim=0.0
    for i in range(lt_obj1.length):
        qt=lt_obj1.term[i]
        localSim=0.0
        # compute p(t|Df)
        w=get_mapping_prob(qt,lucene_obj)
        for f in w:
            len_d_f = entityObj.lengths[f]
            tf_t_d_f = entityObj.term_freqs[f].get(qt,0)
            tf_t_C_f = lucene_obj.get_coll_termfreq(qt, f)
            
            tempSim=get_dirichlet_prob(tf_t_d_f, len_d_f, tf_t_C_f, len_C_f[f], mu[f])
            # compute f(p(t1|De),p(t2|De)...) 
            
            localSim+=w[f]*tempSim
            
        if localSim>0.0:
           totalSim+=math.log(localSim)
    return totalSim

def sdmSim(queryObj,entityObj,field,lucene_obj):
    ft=fo=fu=0.0
    len_C_f = lucene_obj.get_coll_length(field)
    mu=lucene_obj.get_avg_len(field)
    
    ft=lmSim(queryObj.contents_obj,entityObj,field,None,lucene_obj)
    if LAMBDA_O>0:
       for bigram_pair in queryObj.bigrams:
           bigram=bigram_pair[0]+' '+bigram_pair[1]
           tf,cf=lucene_obj.get_coll_bigram_freq(bigram,field,True,0,entityObj.title)
           ptd=get_dirichlet_prob(tf,entityObj.length,cf,len_C_f,mu)
           if ptd>0:
              fo+=math.log(ptd)
    if LAMBDA_U>0:
       for bigram_pair in queryObj.bigrams:
           bigram=bigram_pair[0]+' '+bigram_pair[1]
           tf,cf=lucene_obj.get_coll_bigram_freq(bigram,field,False,6,entityObj.title)
           ptd=get_dirichlet_prob(tf,entityObj.length,cf,len_C_f,mu)
           if ptd>0:
              fu+=math.log(ptd)
    score=LAMBDA_T*ft+LAMBDA_O*fo+LAMBDA_U*fu
    return score
    
def fsdmSim(queryObj,entityObj,lucene_obj):
    fields=LIST_F
    
    len_C_f={}
    mu={}
    for f in LIST_F:
        len_C_f[f]=lucene_obj.get_coll_length(f)
        mu[f]=lucene_obj.get_avg_len(f)
        
    ft=fo=fu=0.0
    # w is a dict of weights for each field
    # compute ft
    for t in queryObj.contents_obj.term:
        w=get_mapping_prob(t,lucene_obj)
        ft_temp=0.0
        for field in w:
            tf=entityObj.term_freqs[field].get(t,0)
            cf=lucene_obj.get_coll_termfreq(t, field)            
            ptd=get_dirichlet_prob(tf,entityObj.lengths[field],cf,len_C_f[field],mu[field])
            if ptd>0:
               ft_temp+=ptd*w[field]
        if ft_temp>0:
           ft+=math.log(ft_temp)
    # compute fo
    if LAMBDA_O>0:
       for bigram_pair in queryObj.bigrams:
           bigram=bigram_pair[0]+' '+bigram_pair[1]
           w=get_mapping_prob(bigram,lucene_obj,ordered=True,slop=0)
           fo_temp=0.0
           for field in w:
               tf,cf=lucene_obj.get_coll_bigram_freq(bigram,field,True,0,entityObj.title)
               ptd=get_dirichlet_prob(tf,entityObj.lengths[field],cf,len_C_f[field],mu[field])
               if ptd>0:
                  fo_temp+=ptd*w[field]
           if fo_temp>0:
              fo+=math.log(fo_temp)
    # compute fu
    if LAMBDA_U>0:
       for bigram_pair in queryObj.bigrams:
           bigram=bigram_pair[0]+' '+bigram_pair[1]
           w=get_mapping_prob(bigram,lucene_obj,ordered=False,slop=6)
           fu_temp=0.0
           for field in w:
               tf,cf=lucene_obj.get_coll_bigram_freq(bigram,field,False,6,entityObj.title)
               ptd=get_dirichlet_prob(tf,entityObj.lengths[field],cf,len_C_f[field],mu[field])
               if ptd>0:
                  fu_temp+=ptd*w[field]
           if fu_temp>0:
              fu+=math.log(fu_temp)
    '''
    if queryObj.contents_obj.length>1:
       ft/=queryObj.contents_obj.length
       fo/=(queryObj.contents_obj.length-1)
       fu/=(queryObj.contents_obj.length-1)
    '''
    score=LAMBDA_T*ft+LAMBDA_O*fo+LAMBDA_U*fu
    return score
    
def get_mapping_prob(t,lucene_obj,ordered=True,slop=0):
    """
    Computes PRMS field mapping probability.
        p(f|t) = P(t|f)P(f) / sum_f'(P(t|C_{f'_c})P(f'))

    :param t: str
    :param coll_termfreq_fields: {field: freq, ...}
    :return Dictionary {field: prms_prob, ...}
    """
    fields=LIST_F
    
    if len(fields)==1:
       # for sdm and lm
       return {fields[0]:1.0}
    
    is_bigram=True if t.find(' ')>-1 else False     
    #find cache
    item=lucene_obj.get_mapping_prob_cached(t,ordered,slop)
    if item is not None:
       return item['weights']
          
    coll_termfreq_fields={}
    
    for f in fields:
        if is_bigram==False:
           coll_termfreq_fields[f]=lucene_obj.get_coll_termfreq(t, f)
        else:
           coll_termfreq_fields[f]=lucene_obj.get_coll_bigram_freq(t,f,ordered,slop,'NONE')[1]

    total_field_freq=lucene_obj.get_total_field_freq(fields)
    # calculates numerators for all fields: P(t|f)P(f)
    numerators = {}
    for f in fields:
        p_t_f = coll_termfreq_fields[f] / lucene_obj.get_coll_length(f)
        p_f = lucene_obj.get_doc_count(f) / total_field_freq
        p_f_t = p_t_f * p_f
        if p_f_t > 0:
           numerators[f] = p_f_t
        else:
           numerators[f]=0

    # calculates denominator: sum_f'(P(t|C_{f'_c})P(f'))
    denominator = sum(numerators.values())

    mapping_probs = {}
    if denominator > 0:  # if the term is present in the collection
       for f in numerators:
           mapping_probs[f] = numerators[f] / denominator
           
    lucene_obj.insert_mapping_prob_cached(t,ordered,slop,mapping_probs)
    return mapping_probs
    
def fsdm_sas(queryObj,entityObj,structure,lucene_handler):
    if len(entityObj.categories)==0:
       return NEGATIVE_INFINITY
    D=structure.cat_dag
    lucene_cat=lucene_handler['category_corpus']
    lucene_doc=lucene_handler['first_pass']
    
    len_d_f=entityObj.lengths
    
    sum_score=0.0
    max_score=NEGATIVE_INFINITY
    len_C_f={}
    sum_ptc={}
    mu={}
    
    # prepare field weights
    w={} # (term,ordered)={field:field_weight}
    for t in queryObj.contents_obj.term:
        w[(t,True)]=get_mapping_prob(t,lucene_doc)
    for bigram_pair in queryObj.bigrams:
        bigram=bigram_pair[0]+' '+bigram_pair[1]
        w[(bigram,True)]=get_mapping_prob(bigram,lucene_doc,ordered=True,slop=0)
    for bigram_pair in queryObj.bigrams:
        bigram=bigram_pair[0]+' '+bigram_pair[1]
        w[(bigram,False)]=get_mapping_prob(bigram,lucene_doc,ordered=False,slop=6)
        
    for field in LIST_F:
        len_C_f[field]=lucene_doc.get_coll_length(field)
        mu[field]=lucene_doc.get_avg_len(field)
        sum_ptc[('T',field)]=[0.0 for i in range(queryObj.contents_obj.length)]
        sum_ptc[('O',field)]=[0.0 for i in range(len(queryObj.bigrams))]
        sum_ptc[('U',field)]=[0.0 for i in range(len(queryObj.bigrams))]
        
    curPath=[]

    def smooth_path(cat,path_len,alpha_t,sum_nominator):
        nonlocal D,curPath,sum_ptc,cnt_path
        nonlocal max_score_p_cat,max_score
        nonlocal lucene_cat,lucene_doc
        
        if cnt_path>TOP_PATH_NUM_PER_CAT:
           return
        # the following is end condition
        if path_len==LIMIT_SAS_PATH_LENGTH or len(D[cat])==0:
           # compute score
           cnt_path+=1
           if alpha_t==ALPHA_SAS:
              return           
           # TAKE CARE OF COF !
           #cof=(1-ALPHA_SAS)/(ALPHA_SAS-alpha_t)
           cof=0.003
           score_p=0.0
           # for individual query terms
           ft_p=0.0
           for j in range(queryObj.contents_obj.length):
               term=queryObj.contents_obj.term[j]
               ptd=0.0
               for f in w[(term,True)]:
                   tf_d_f=entityObj.term_freqs[f].get(term,0.0)
                   cf_f = lucene_doc.get_coll_termfreq(term, f)
                   ptc_f=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                   Dt=mu[f]*ptc_f
                   Nt=mu[f]
                   if f not in NO_SMOOTHING_LIST:
                      Dt+=cof*sum_ptc[('T',f)][j]
                      Nt+=cof*sum_nominator[f]
                   ptd_f=(tf_d_f+Dt)/(len_d_f[f]+Nt) if len_d_f[f]+Nt>0 else 0.0
                   ptd+=w[(term,True)][f]*ptd_f
               if ptd>0:
                  ft_p+=math.log(ptd)
           # for ordered bigrams
           fo_p=0.0
           if LAMBDA_O>0:
              for j in range(len(queryObj.bigrams)):
                  bigram=queryObj.bigrams[j][0]+' '+queryObj.bigrams[j][1]
                  ptd=0.0
                  for f in w[(bigram,True)]:
                      tf_d_f,cf_f=lucene_doc.get_coll_bigram_freq(bigram,f,True,0,entityObj.title)
                      ptc_f=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                      Dt=mu[f]*ptc_f
                      Nt=mu[f]
                      if f not in NO_SMOOTHING_LIST:
                         Dt+=cof*sum_ptc[('O',f)][j]
                         Nt+=cof*sum_nominator[f]
                      ptd_f=(tf_d_f+Dt)/(len_d_f[f]+Nt) if len_d_f[f]+Nt>0 else 0.0
                      ptd+=w[(bigram,True)][f]*ptd_f
                  if ptd>0:
                     fo_p+=math.log(ptd)*w[(bigram,True)][f]           
           # for unordered bigrams
           fu_p=0.0
           if LAMBDA_U>0:
              for j in range(len(queryObj.bigrams)):
                  bigram=queryObj.bigrams[j][0]+' '+queryObj.bigrams[j][1]
                  ptd=0.0
                  for f in w[(bigram,False)]:
                      tf_d_f,cf_f=lucene_doc.get_coll_bigram_freq(bigram,f,False,6,entityObj.title)
                      ptc_f=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                      Dt=mu[f]*ptc_f
                      Nt=mu[f]
                      if f not in NO_SMOOTHING_LIST:
                         Dt+=cof*sum_ptc[('U',f)][j]
                         Nt+=cof*sum_nominator[f]
                      ptd_f=(tf_d_f+Dt)/(len_d_f[f]+Nt) if len_d_f[f]+Nt>0 else 0.0
                      ptd+=w[(bigram,False)][f]*ptd_f
                  if ptd>0:
                     fu_p+=math.log(ptd)*w[(bigram,False)][f]           
           # end computing feature function
           score_p=LAMBDA_T*ft_p+LAMBDA_O*fo_p+LAMBDA_U*fu_p
           if score_p>max_score_p_cat:
              max_score_p_cat=score_p
           return
           
        # maintain useful temporary variables
        # current node is cat
        cat_corpus,docID=lucene_cat.findDoc(cat,'category',True)
        bak_sum_ptc=sum_ptc.copy()
        if cat_corpus is not None:
           # maintain
           cnt_doc_corpus=int(cat_corpus['num_articles'])
           for f in LIST_F:
               if f in NO_SMOOTHING_LIST:
                  continue
               # get category corpus
               term_freq_c=lucene_cat.get_term_freq(docID,f,True)
               len_c=sum(term_freq_c.values())
               mu_c=len_c/cnt_doc_corpus if cnt_doc_corpus>0 else 0.0
               sum_nominator[f]+=alpha_t*mu_c       
               # maintain individual query terms
               for j in range(queryObj.contents_obj.length):
                   term=queryObj.contents_obj.term[j]
                   cf_c=term_freq_c.get(term,0.0)     
                   ptc_f=cf_c/len_c if len_c>0 else -1    
                   if ptc_f>-1:  
                      sum_ptc[('T',f)][j]+=(alpha_t*ptc_f*mu_c)                     
               # maintain ordered bigrams
               if LAMBDA_O>0:
                  for j in range(len(queryObj.bigrams)):
                      bigram=queryObj.bigrams[j][0]+' '+queryObj.bigrams[j][1]
                      cf_c,cf_cc=lucene_cat.get_coll_bigram_freq(bigram,f,True,0,cat,field_cache='category')
                      ptc_f=cf_c/len_c if len_c>0 else -1
                      if ptc_f>-1:
                         sum_ptc[('O',f)][j]+=(alpha_t*ptc_f*mu_c)
               # maintain unordered bigrams
               if LAMBDA_U>0:
                  for j in range(len(queryObj.bigrams)):
                      bigram=queryObj.bigrams[j][0]+' '+queryObj.bigrams[j][1]
                      cf_c,cf_cc=lucene_cat.get_coll_bigram_freq(bigram,f,False,6,cat,field_cache='category')
                      ptc_f=cf_c/len_c if len_c>0 else -1
                      if ptc_f>-1:
                         sum_ptc[('U',f)][j]+=(alpha_t*ptc_f*mu_c)               
        cnt=0
        for child in iter(D[cat]):
            cnt+=1
            if cnt>TOP_CATEGORY_NUM:
               break
            if child in D:
               curPath.append(child)
               smooth_path(child,path_len+1,alpha_t*ALPHA_SAS,sum_nominator)
               curPath.pop()
               sum_ptc=bak_sum_ptc.copy()
    # end of function smooth_path
    
    for cat in entityObj.categories[:TOP_CATEGORY_NUM]:
        if cat not in D:
           continue
        max_score_p_cat=NEGATIVE_INFINITY     
        cnt_path=0
        smooth_path(cat,1,ALPHA_SAS,{f:0.0 for f in LIST_F})
        
        if max_score_p_cat>NEGATIVE_INFINITY:
           sum_score+=max_score_p_cat
        if max_score_p_cat>max_score:
           max_score=max_score_p_cat
               
    return max_score
# ============================

def prms_sas(queryObj,entityObj,structure,lucene_handler):
    if len(entityObj.categories)==0:
       return NEGATIVE_INFINITY
    D=structure.cat_dag
    lucene_cat=lucene_handler['category_corpus']
    lucene_doc=lucene_handler['first_pass']
    
    len_d_f=entityObj.lengths
    
    sum_score=0.0
    max_score=NEGATIVE_INFINITY
    len_C_f={}
    mlm_weights={}
    sum_ptc={}
    mu={}
    
    for t in queryObj.contents_obj.term:
        mlm_weights[t]=get_mapping_prob(t,lucene_doc)
        
    for field in LIST_F:
        len_C_f[field]=lucene_doc.get_coll_length(field)
        mu[field]=lucene_doc.get_avg_len(field)
        sum_ptc[field]=[0.0 for i in range(queryObj.contents_obj.length)]

    curPath=[]

    def smooth_path(cat,path_len,alpha_t,sum_nominator):
        nonlocal D,curPath,sum_ptc,cnt_path
        nonlocal max_score_p_cat,max_score
        nonlocal lucene_cat,lucene_doc
        
        if cnt_path>TOP_PATH_NUM_PER_CAT:
           return
        # the following is end condition
        if path_len==LIMIT_SAS_PATH_LENGTH or len(D[cat])==0:
           # compute score
           cnt_path+=1
           if alpha_t==ALPHA_SAS:
              return      
           #cof=(1-ALPHA_SAS)/(ALPHA_SAS-alpha_t)
           cof=0.003
           #cof=1
           score_p=0.0
           for j in range(queryObj.contents_obj.length):
               term=queryObj.contents_obj.term[j]
               ptd=0.0
               for f in mlm_weights[term]:
                   tf_d_f=entityObj.term_freqs[f].get(term,0.0)
                   cf_f = lucene_doc.get_coll_termfreq(term, f)
                   ptc_doc=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                   if f in NO_SMOOTHING_LIST:
                      ptd_f=(tf_d_f+mu[f]*ptc_doc)/(len_d_f[f]+mu[f]) if (len_d_f[f]+mu[f])>0 else 0.0                                   
                   else:
                      ptd_f=(tf_d_f+mu[f]*ptc_doc+cof*sum_ptc[f][j])/(len_d_f[f]+mu[f]+cof*sum_nominator[f]) if (len_d_f[f]+mu[f]+cof*sum_nominator[f])>0 else 0.0             
                   ptd+=mlm_weights[term][f]*ptd_f
               if ptd>0:
                  score_p+=math.log(ptd)
           if score_p>max_score_p_cat:
              max_score_p_cat=score_p
           return
           
        # maintain useful temporary variables
        d,docID=lucene_cat.findDoc(cat,'category',True)
        bak_sum_ptc=sum_ptc.copy()
        
        cnt_doc_corpus=0
        if d is not None:
           # maintain
           cnt_doc_corpus=int(d['num_articles'])
           for f in LIST_F:
               if f in NO_SMOOTHING_LIST:
                  continue
               # get category corpus
               term_freq=lucene_cat.get_term_freq(docID,f,True)
               len_c=sum(term_freq.values())
               mu_c=len_c/cnt_doc_corpus if cnt_doc_corpus>0 else 0.0
               sum_nominator[f]+=alpha_t*mu_c
               
               for j in range(queryObj.contents_obj.length):
                   term=queryObj.contents_obj.term[j]
                   tf_c=term_freq.get(term,0.0)     
                   ptc=tf_c/len_c if len_c>0 else -1    
                   if ptc>-1:  
                      sum_ptc[f][j]+=(alpha_t*mu_c*ptc)
                                     
        cnt=0
        for child in iter(D[cat]):
            if child in D:
               curPath.append(child)
               smooth_path(child,path_len+1,alpha_t*ALPHA_SAS,sum_nominator)
               curPath.pop()
               sum_ptc=bak_sum_ptc.copy()
            cnt+=1
            if cnt>TOP_CATEGORY_NUM:
               break
    # end of function smooth_path
    
    for cat in entityObj.categories[:TOP_CATEGORY_NUM]:
        if cat not in D:
           continue
        max_score_p_cat=NEGATIVE_INFINITY     
        cnt_path=0
        smooth_path(cat,1,ALPHA_SAS,{f:0.0 for f in LIST_F})
        
        if max_score_p_cat>NEGATIVE_INFINITY:
           sum_score+=max_score_p_cat
        if max_score_p_cat>max_score:
           max_score=max_score_p_cat
               
    return max_score    
    
def mlm_sas(queryObj,entityObj,structure,lucene_handler):
    if len(entityObj.categories)==0:
       return NEGATIVE_INFINITY
    D=structure.cat_dag
    lucene_cat=lucene_handler['category_corpus']
    lucene_doc=lucene_handler['first_pass']
    
    len_d_f=entityObj.lengths
    
    sum_score=0.0
    max_score=NEGATIVE_INFINITY
    len_C_f={}
    mlm_weights={}
    sum_ptc={}
    mu={}
    for field in LIST_F:
        len_C_f[field]=lucene_doc.get_coll_length(field)
        mu[field]=lucene_doc.get_avg_len(field)
        mlm_weights[field]=1.0/len(LIST_F)
        sum_ptc[field]=[0.0 for i in range(queryObj.contents_obj.length)]
        
    if MODEL_NAME=='mlm-tc':
       mlm_weights={'stemmed_names':0.2,'stemmed_catchall':0.8} if USED_QUERY_VERSION=='stemmed_raw_query' else {'names':0.2,'attributes':0.8} 

    curPath=[]

    def smooth_path(cat,path_len,alpha_t,sum_nominator):
        nonlocal D,curPath,sum_ptc,cnt_path
        nonlocal max_score_p_cat,max_score
        nonlocal lucene_cat,lucene_doc
        
        if cnt_path>TOP_PATH_NUM_PER_CAT:
           return
        # the following is end condition
        if path_len==LIMIT_SAS_PATH_LENGTH or len(D[cat])==0:
           # compute score
           cnt_path+=1
           if alpha_t==ALPHA_SAS:
              return      
           cof=(1-ALPHA_SAS)/(ALPHA_SAS-alpha_t)
           #cof=0.003
           #cof=1
           score_p=0.0
           for j in range(queryObj.contents_obj.length):
               term=queryObj.contents_obj.term[j]
               ptd=0.0
               for f in LIST_F:
                   tf_d_f=entityObj.term_freqs[f].get(term,0.0)
                   cf_f = lucene_doc.get_coll_termfreq(term, f)
                   ptc_doc=cf_f/len_C_f[f] if len_C_f[f]>0 else 0.0
                   if f in NO_SMOOTHING_LIST:
                      ptd_f=(tf_d_f+mu[f]*ptc_doc)/(len_d_f[f]+mu[f]) if (len_d_f[f]+mu[f])>0 else 0.0                                   
                   else:
                      ptd_f=(tf_d_f+mu[f]*ptc_doc+cof*sum_ptc[f][j])/(len_d_f[f]+mu[f]+cof*sum_nominator[f]) if (len_d_f[f]+mu[f]+cof*sum_nominator[f])>0 else 0.0             
                   #ptd_f=(tf_d_f+mu[f]*(ptc_doc+cof*sum_ptc[f][j]))/(len_d_f[f]+mu[f]) if len_d_f[f]+mu[f]>0 else 0.0             
                   '''
                   if tf_d_f>0:
                      ptd_f=(tf_d_f+mu[f]*ptc_doc+cof*sum_ptc[f][j])/(len_d_f[f]+mu[f]+cof*sum_nominator[f]) if len_d_f[f]+mu[f]+cof*sum_nominator[f]>0 else 0.0
                   else:
                      ptd_f=(tf_d_f+mu[f]*ptc_doc)/(len_d_f[f]+mu[f]) if len_d_f[f]+mu[f]>0 else 0.0
                   '''

                   ptd+=mlm_weights[f]*ptd_f
               if ptd>0:
                  score_p+=math.log(ptd)
           if score_p>max_score_p_cat:
              max_score_p_cat=score_p
           return
           
        # maintain useful temporary variables
        d,docID=lucene_cat.findDoc(cat,'category',True)
        bak_sum_ptc=sum_ptc.copy()
        
        cnt_doc_corpus=0
        if d is not None:
           # maintain
           cnt_doc_corpus=int(d['num_articles'])
           for f in LIST_F:
               
               if f in NO_SMOOTHING_LIST:
                  continue
              
               # get category corpus
               term_freq=lucene_cat.get_term_freq(docID,f,True)
               len_c=sum(term_freq.values())
               mu_c=len_c/cnt_doc_corpus if cnt_doc_corpus>0 else 0.0
               sum_nominator[f]+=alpha_t*mu_c
               #sum_nominator[f]+=alpha_t*mu_c/cnt_doc_corpus
               #sum_nominator[f]+=mu_c/(path_len+1)               
               '''
               # add this
               mu_cc_f=lucene_cat.get_avg_len(field)               

               for j in range(queryObj.contents_obj.length):
                   term=queryObj.contents_obj.term[j]
                   tf_c=term_freq.get(term,0.0)     
                   cf_cc_f = lucene_cat.get_coll_termfreq(term, f)
                   len_cc_f=lucene_cat.get_coll_length(field)
                   ptc=get_dirichlet_prob(tf_c,len_c,cf_cc_f,len_cc_f,mu_cc_f)
                   if ptc>-1:  
                      sum_ptc[f][j]+=(alpha_t*mu_c*ptc)   
               '''
               
               for j in range(queryObj.contents_obj.length):
                   term=queryObj.contents_obj.term[j]
                   tf_c=term_freq.get(term,0.0)     
                   ptc=tf_c/len_c if len_c>0 else -1    
                   if ptc>-1:  
                      #sum_ptc[f][j]+=(alpha_t*mu_c*ptc/cnt_doc_corpus) 
                      sum_ptc[f][j]+=(alpha_t*mu_c*ptc)
                      #sum_ptc[f][j]+=(alpha_t*ptc) 
                      #sum_ptc[f][j]+=(ptc/(path_len+1))
                                     
        cnt=0
        for child in iter(D[cat]):
            if child in D:
               curPath.append(child)
               smooth_path(child,path_len+1,alpha_t*ALPHA_SAS,sum_nominator)
               curPath.pop()
               sum_ptc=bak_sum_ptc.copy()
            cnt+=1
            if cnt>TOP_CATEGORY_NUM:
               break
    # end of function smooth_path
    
    for cat in entityObj.categories[:TOP_CATEGORY_NUM]:
        if cat not in D:
           continue
        max_score_p_cat=NEGATIVE_INFINITY     
        cnt_path=0
        smooth_path(cat,1,ALPHA_SAS,{f:0.0 for f in LIST_F})
        
        if max_score_p_cat>NEGATIVE_INFINITY:
           sum_score+=max_score_p_cat
        if max_score_p_cat>max_score:
           max_score=max_score_p_cat
               
    return max_score    
#===========================================
def lm_sas(queryObj,entityObj,structure,lucene_handler,mongoObj,field):
    if len(entityObj.categories)==0:
       return NEGATIVE_INFINITY
    D=structure.cat_dag
    lucene_cat=lucene_handler['category_corpus']
    lucene_doc=lucene_handler['first_pass']
    
    termList=entityObj.term_freq
    len_d=entityObj.length
    
    sum_score=0.0
    max_score=NEGATIVE_INFINITY
    len_C_f = lucene_doc.get_coll_length(field)
    mu_d=lucene_doc.get_avg_len(field)

    curPath=[]
    sum_ptc=[0.0 for i in range(queryObj.contents_obj.length)]
    
    def smooth_path(cat,path_len,alpha_t,sum_nominator):
        nonlocal D,curPath,sum_ptc,cnt_path
        nonlocal max_score_p_cat,max_score
        nonlocal lucene_cat,lucene_doc
        
        #print (cat)
        if cnt_path>TOP_PATH_NUM_PER_CAT:
           return
        if path_len==LIMIT_SAS_PATH_LENGTH or len(D[cat])==0:
           # compute score
           cnt_path+=1
           if alpha_t==ALPHA_SAS:
              return       
           cof=(1-ALPHA_SAS)/(ALPHA_SAS-alpha_t)
           #cof=0.003
           #cof=1
           #cof=0.3
           # 0.3 for DBpedia
           score_p=0.0
           for j in range(queryObj.contents_obj.length):
                term=queryObj.contents_obj.term[j]
                tf_d=entityObj.term_freq.get(term,0.0)
                tf_t_C_f = lucene_doc.get_coll_termfreq(term, field)
                ptc_doc=tf_t_C_f/len_C_f if len_C_f>0 else 0.0
                ptd=(tf_d+mu_d*ptc_doc+cof*sum_ptc[j])/(len_d+mu_d+cof*sum_nominator) if len_d+mu_d+cof*sum_nominator>0 else 0.0
                #print ('%s\t%f\t%f'%(term,sum_ptc[j],sum_nominator))
                '''
                if tf_d>0 and sum_ptc[j]>0:
                   ptd=(tf_d+mu_d*ptc_doc+cof*sum_ptc[j])/(len_d+mu_d+cof*sum_nominator) if len_d+mu_d+cof*sum_nominator>0 else 0.0
                else:
                   ptd=(tf_d+mu_d*ptc_doc)/(len_d+mu_d) if len_d+mu_d>0 else 0.0
                   #ptd=0 will impact performance for lm on all datasets
                '''
                if ptd>0:
                   score_p+=math.log(ptd)
           if score_p>max_score_p_cat:
              max_score_p_cat=score_p
           return
           
        # maintain useful temporary variables
        d,docID=lucene_cat.findDoc(cat,'category',True)
        bak_sum_ptc=sum_ptc[:]
        if d is not None:
           # maintain
           
           term_freq=lucene_cat.get_term_freq(docID,field,True)
           len_c=sum(term_freq.values())
           cnt_doc_corpus=int(d['num_articles'])
           mu_c=len_c/cnt_doc_corpus if cnt_doc_corpus>0 else 0.0
           sum_nominator+=alpha_t*mu_c         

           #print ('find %s, len_c=%d, mu_c=%f'%(cat,len_c,mu_c))
           
           for j in range(queryObj.contents_obj.length):
               term=queryObj.contents_obj.term[j]
               tf_c=term_freq.get(term,0.0)     
               ptc=tf_c/len_c if len_c>0 else -1    
               if ptc>-1:  
                  sum_ptc[j]+=(alpha_t*ptc*mu_c)                     
        cnt=0
        for child in iter(D[cat]):
            cnt+=1
            if cnt>TOP_CATEGORY_NUM:
               break
            if child in D:
               curPath.append(child)
               smooth_path(child,path_len+1,alpha_t*ALPHA_SAS,sum_nominator)
               curPath.pop()
               sum_ptc=bak_sum_ptc[:]
    # end of function smooth_path
      
    for cat in entityObj.categories[:TOP_CATEGORY_NUM]:
        if cat not in D:
           continue
        #print ('---')
        max_score_p_cat=NEGATIVE_INFINITY     
        cnt_path=0
        #smooth_path(cat,1,1.0,0.0)
        smooth_path(cat,1,ALPHA_SAS,0.0)
        if max_score_p_cat>NEGATIVE_INFINITY:
           sum_score+=max_score_p_cat
        if max_score_p_cat>max_score:
           max_score=max_score_p_cat
        #print ('cnt_path=%d'%(cnt_path))           
    return max_score

# ============================
def scoreWikiTree(queryObj,T_obj,lucene_obj,field):
    curPath=[]
    bestPath=[]
    maxScore=NEGATIVE_INFINITY
    T=T_obj.T
    mu=lucene_obj.get_avg_len(field)
    len_C = lucene_obj.get_coll_length(field)
    sum_w_tf_ug=[0.0 for i in range(queryObj.contents_obj.length)]
    sum_w_tf_ob=[0.0 for i in range(len(queryObj.bigrams))]
    sum_w_tf_ub=[0.0 for i in range(len(queryObj.bigrams))]  
    
    def scorePath(v,sum_w_len,len_path):
        # v:node sum_w_tf:sum of weighted tf, sum_w_len:sum of weighted doc len
        nonlocal T_obj,T,lucene_obj,queryObj
        nonlocal field,maxScore
        nonlocal curPath,bestPath,sum_w_tf_ug,sum_w_tf_ob,sum_w_tf_ub
        # slow  revise traverse
        
        if v==-1 or len_path>LIMIT_D_PATH_LENGTH:
           score_T=score_U=score_O=0.0
           for i in range(queryObj.contents_obj.length):
               term=queryObj.contents_obj.term[i]
               cf=lucene_obj.get_coll_termfreq(term,field)
               score_i=get_dirichlet_prob(sum_w_tf_ug[i], sum_w_len, cf, len_C, mu)
               if score_i>0:
                  score_T+=math.log(score_i)
           for i in range(len(queryObj.bigrams)):
               bigram=queryObj.bigrams[i][0]+' '+queryObj.bigrams[i][1]
               cf=lucene_obj.get_coll_bigram_freq(bigram,field,True,0,T_obj.title,'title')[1]
               score_i=get_dirichlet_prob(sum_w_tf_ob[i], sum_w_len, cf, len_C, mu)
               if score_i>0:
                  score_O+=math.log(score_i)
                  
           for i in range(len(queryObj.bigrams)):
               bigram=queryObj.bigrams[i][0]+' '+queryObj.bigrams[i][1]
               cf=lucene_obj.get_coll_bigram_freq(bigram,field,False,6,T_obj.title,'title')[1]
               score_i=get_dirichlet_prob(sum_w_tf_ub[i], sum_w_len, cf, len_C, mu)
               if score_i>0:
                  score_U+=math.log(score_i)   
           score=LAMBDA_T*score_T+LAMBDA_O*score_O+LAMBDA_U*score_U
           if score==0:
              score=NEGATIVE_INFINITY           
           if score>maxScore:
              maxScore=score
              bestPath=curPath.copy()
        else:
           content=' '.join(T[v][field])
           T[v]['list_term_object']=List_Term_Object(content,True,' ',None,None,is_bigram_used=True)
           lto=T[v]['list_term_object']
           bak_ug=sum_w_tf_ug.copy()
           bak_ob=sum_w_tf_ob.copy()
           bak_ub=sum_w_tf_ub.copy()
           
           for i in range(queryObj.contents_obj.length):
               term=queryObj.contents_obj.term[i]
               tf=lto.term_freq.get(term,0)
               sum_w_tf_ug[i]=sum_w_tf_ug[i]*ALPHA_D+tf
           for i in range(len(queryObj.bigrams)):
               bigram=queryObj.bigrams[i][0]+' '+queryObj.bigrams[i][1]
               tf=lto.bigram_freq.get(bigram,0)
               sum_w_tf_ob[i]=sum_w_tf_ob[i]*ALPHA_D+tf
           for i in range(len(queryObj.bigrams)):
               bigram=queryObj.bigrams[i]
               term1,term2=bigram
               p2=tf=0
               # can be optimized via suffix array
               for p1 in range(queryObj.contents_obj.length):
                   if queryObj.contents_obj.term[p1] not in [term1,term2]:
                      continue
                   for p2 in range(p1+1,queryObj.contents_obj.length):
                       if p2-p1-1>6:
                          break
                       elif queryObj.contents_obj.term[p2] in [term1,term2]:
                            tf+=1     
               sum_w_tf_ub[i]=sum_w_tf_ub[i]*ALPHA_D+tf  
               
           if len(T[v]['child'])>0:
              for c in T[v]['child']:
                  scorePath(c,sum_w_len*ALPHA_D+lto.length,len_path+1)
           else:
              scorePath(-1,sum_w_len*ALPHA_D+lto.length,len_path+1)
           sum_w_tf_ug=bak_ug.copy()
           sum_w_tf_ob=bak_ob.copy()
           sum_w_tf_ub=bak_ub.copy()
    # ----------------------------------------------------    
    for v in T[1]['child']:    
        sum_w_tf_ug=[0.0 for i in range(queryObj.contents_obj.length)]
        sum_w_tf_ob=[0.0 for i in range(len(queryObj.bigrams))]
        sum_w_tf_ub=[0.0 for i in range(len(queryObj.bigrams))]    
        scorePath(v,0.0,0)
    return maxScore


def scoreQueryGraph(queryObj,entityObj,lucene_obj,mongoObj):
    field='catchall'

    conn=mongoObj.conn_mbpo
    rel_preds=set(queryObj.query_predicates.keys())

    #print (entityObj.title)
    sum_score=0.0
    sum_qe_scores=sum(queryObj.query_entities.values())
    # dest_entity in uri
    
    reachable=False
    
    def findPath(v,path_len,prod_p):
        nonlocal dest_entity, curPath
        nonlocal maxScore, score_qc, se, sum_qe_scores,reachable
       
        if path_len==MAX_CHAIN_LENGTH or v==dest_entity:
           if v==dest_entity:
              reachable=True
           score_path=0.0
           if v==dest_entity:
              score_path=1.0
 
           score=0.6*score_path+0.2*score_qc+0.2*prod_p       
           if score>maxScore:
              maxScore=score
           return
           
         # expand graph
        list_item=conn.find({'uri':v})
        if list_item is None:
           return
              
        for item in list_item:
            pred=item['property']
            target_en_uri=item['value']
            # newly added
            if target_en_uri in curPath:
               continue
            if pred not in queryObj.query_predicates:
               continue
            prob_r=queryObj.query_predicates[pred]
            curPath.append(target_en_uri)
            findPath(target_en_uri,path_len+1,prod_p*prob_r)
            curPath.pop()
                
    dest_entity=entityObj.uri
    curPath=[]
    df_f=lucene_obj.get_doc_count(field)
    for entity in queryObj.query_entities:
        uri='<http://dbpedia.org/resource/'+entity+'>'
        del curPath[:]
        
        # compute prob(q_e|candidate)
        d,docID=lucene_obj.findDoc(entity,'title',True)
        if d is None or docID is None:
           continue
        se=queryObj.query_entities[entity]
        title=entity.lower()
        term_freq=lucene_obj.get_term_freq(docID,field,False)
        tf=1 if title in term_freq else 0
        df_e_f=lucene_obj.get_doc_freq(title, field)
        score_qc=0.9*tf+0.1*(df_e_f/df_f)
        # --------------------------------------
        
        maxScore=0.0
        findPath(uri,0,1.0)
        
        #sum_score+=(queryObj.query_entities[entity]/sum_qe_scores)*maxScore
        sum_score=max(sum_score,maxScore)
    
    return sum_score,reachable
        