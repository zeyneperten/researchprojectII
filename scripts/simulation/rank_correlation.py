import numpy as np
from scipy.stats import binom

def rankseq(s1,s2):

    #compute rank order correlation between sequences

    # set things straight
    s1=np.array(s1).flatten()
    s2=np.array(s2).flatten()
    l1=len(s1)
    l2=len(s2)
    
    #difference matrix
    d=np.ones((l1,1))*s2 - (np.ones((l2,1))*s1).transpose()
  
    # binary identity matrix
    d=(d==0)
  

    # make s0 the shorter sequence
    s=s1
    s0=s2
    l0=l2
    ln=l1
    if l1<l2:
        s=s2;
        s0=s1;
        l0=l1
        ln=l2
        d=d.transpose()


 
        
    #compute cell overlap (neurons contained in both)
    minseq=s[np.where(np.sum(d,axis=1)>0)[0]];
    lm=len(minseq)
  
    # delete neurons from the shorter sequence that are not in the minimal
    # sequence
    #
    
    d0=np.ones((l0,1))*minseq - (np.ones((lm,1))*s0).transpose()
    d0=(d0==0)
    s0=s0[np.sum(d0,axis=1)>0]
    l0=len(s0)
  
  
    #find ordinal rank in the shorter sequence
    dd=np.ones((lm,1))*s0 - (np.ones((l0,1))*minseq).transpose()
  
    #compute spearmans r
    if len(dd)>1:
        ids=np.argmin(np.abs(dd),axis=0)
        
        rc = np.corrcoef(np.arange(len(ids)),ids)[0,1]
        ln=len(ids)
    else:
        rc=np.nan;
        ln=np.nan
  
   
    
    return rc, ln

    
def allmot(seqs,nrm):

    nseqs=len(seqs)

    narr=np.array(nrm)[:,0]

    corrmat=np.zeros((nseqs,nseqs))
    zmat=np.zeros((nseqs,nseqs))
    bmat=np.zeros((nseqs,nseqs))
    pval=np.zeros(nseqs)
    nsig=np.zeros(nseqs)
    
    for ns in range(nseqs):

        s1=seqs[ns]

        zmat[ns,ns]=np.nan
        bmat[ns,ns]=np.nan
        for ms in range(ns+1,nseqs):

            

            s2=seqs[ms]

            rc,ln=rankseq(s1,s2)
            

            if ln>=50:
                mns=nrm[-1]
            else:
                whichone=np.array(np.where(ln==narr)).flatten()
                if len(whichone)==0:
                    mns=np.empty(4)
                    mns[:]=np.nan
                else:
                    mns=nrm[whichone[0]]
                    
                    
            ztmp=(rc-mns[1])/mns[2]
            #print(ns,mns,ztmp)
            corrmat[ns,ms]=rc
            corrmat[ms,ns]=rc

            zmat[ns,ms]=ztmp
            zmat[ms,ns]=ztmp
            bmat[ns,ms]=1.*(ztmp>mns[3])
            bmat[ms,ns]=1.*(ztmp>mns[3])

        nsig[ns] = np.nansum(bmat[ns,:])
        pval[ns] = 1-binom.cdf(nsig[ns],nseqs-1,.05)# i will change pvalue from .05 to 0.01 for merging(hamed 02.02.2023)


    rep_index = nsig/np.std(nsig)


    return rep_index, nsig, pval, bmat, zmat, corrmat