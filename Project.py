#!/usr/bin/env python
# coding: utf-8

# In[537]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
data = pd.read_csv(r'C:\Users\hp\Downloads\combined_hourly_16_17_18.csv')
print (data.head())
data.dropna(subset = ["bc1"], inplace=True)
data.dropna(subset = ["bc2"], inplace=True)
data.dropna(subset = ["bc3"], inplace=True)
data.dropna(subset = ["bc4"], inplace=True)
data.dropna(subset = ["bc5"], inplace=True)
data.dropna(subset = ["bc6"], inplace=True)
data.dropna(subset = ["bc7"], inplace=True)
data= data.reset_index(drop = True)
print((data["bc1"]))
print(data["bc1"][0])
for i in range (335082):
  if (data["stn"][i]=="pne" and data["date"][i]=="01-10-2018"):
    print(i)
    break
print(data['bc1'][i]) 
for j in range (335082):
  if (data["stn"][j]=="pne" and data["date"][j]=="01-12-2018"): 
    print(j)
    break
print(data['bc1'][j]) 
print(j-i)


# In[538]:


import numpy as np
abs = [data["bc1"]*.01847 , data["bc2"]*0.01454 , data["bc3"]*0.01314 , data["bc4"]*0.01158 , data["bc5"]*0.01035 , data["bc6"]*0.00777 , data["bc7"]*0.00719] 
oc = [data["bc1"], data["bc2"] , data["bc3"], data["bc4"] , data["bc5"] , data["bc6"] , data["bc7"]]
l = [ 370 , 470 , 520 , 590 , 660 , 880 , 950 ]
print(l)
print(abs[0][3])

print(l[3])
print(abs)
print(abs[0][10613])
print(abs[0][10614])
print(abs[0][10615])
print (len(abs[0]))
#print(abs[1][55])244069
#abs=np.array(abs)
#abs=abs.reshape(335083,7)
#print(abs)
for i in range (7):
 for j in range (1907,10612):
    print(i,j)
    print(abs[i][j])


# In[539]:


y=[]
for i in range (7):
 for j in range (301429,302702):
   # print(i,j) 
    #print(abs[i][j])
    d=abs[i][j]*l[i]
    y.append(d)
y = np.array(y)
y = y.reshape(7,1273)
y= np.transpose(y)
print(y)
print(np.shape(y))

        


# In[225]:


x=[]
for j in range(7):
    a=pow(l[j],-3)
    x.append(a)
x = np.array(x)
b = np.array([1,1,1,1,1,1,1])
X = np.concatenate([x,b])
X = X.reshape(-1,1)
X = np.transpose(X)
print(X)
X = X.reshape(2,7)
X = np.transpose(X)
print(X)
print(x)


# In[545]:


from sklearn.linear_model import LinearRegression
#Create an object called regressor in the LinearRegression class...
regr = LinearRegression()
#Fit the linear regression model to the training set... We use the fit method
#the arguments of the fit method will be training sets 
score=[]
for k in np.arange(2.0, 8.0, 0.1):
 x=[]
 for n in range(7):
  a=pow(l[n],(1-k))
  x.append(a)
 print(x)
 x = np.array(x)
 b = np.array([1,1,1,1,1,1,1])
 X = np.concatenate([x,b])
 X = X.reshape(-1,1)
 X = np.transpose(X)
 print(X)
 X = X.reshape(2,7)
 X = np.transpose(X)
 print(X)
 for j in range(1273):
    regr.fit(X,y[j])
    score.append([regr.score(X,y[j]),j, k])
print(score)
max_score=[]
max_score.append(max(score))
print(max_score)
index = max_score[0][1]
AAE = max_score[0][2]
print(index,AAE)
#print(regr.coef_, regr.intercept_)

#7 Predicting the Test set results: 
#y_pred= regr.predict(X_test)
#y_pred


# In[8]:


from sklearn.svm import SVR
scr=[]
for k in range(2, 8, 1):
 x=[]
 for n in range(7):
  a=pow(l[n],(1-k))
  x.append(a)
 print(x)
 x = np.array(x)
 b = np.array([1,1,1,1,1,1,1])
 X = np.concatenate([x,b])
 X = X.reshape(-1,1)
 X = np.transpose(X)
 print(X)
 X = X.reshape(2,7)
 X = np.transpose(X)
 print(X)
 for j in range(200):
    regressor = SVR(kernel='linear')
    regressor.fit(X,y[j])
    scr.append([regressor.score(X,y[j]),j, k])
print(scr)
print(max(scr))


# In[546]:


x_new=[]
for n in range(7):
 c=pow(l[n],(1-AAE))
 x_new.append(c)
print(x_new)
x_new = np.array(x_new)
b = np.array([1,1,1,1,1,1,1])
X_new = np.concatenate([x_new,b])
X_new = X_new.reshape(-1,1)
X_new = np.transpose(X_new)
print(X_new)
X_new = X_new.reshape(2,7)
X_new = np.transpose(X_new)
print(X_new)
regr1=LinearRegression()
regr1.fit(X_new,y[index])
y_pred=regr1.predict(X_new)
print(y_pred)
print(y[index])
print(regr1.score(X_new,y[index]))
X_new = np.transpose(X_new)
plt.scatter(X_new[0],y_pred, color = 'brown')
plt.plot(X_new[0],y_pred, color = 'black')
plt.title('Regression fit')
plt.xlabel('l^-3.7')
plt.ylabel('ATN*l')
plt.show()
print(regr1.coef_)
print(regr1.intercept_)


# In[542]:


#abs=np.array(abs)
#abs=abs.reshape(len(abs[:,0]),7)
#print(abs[0])
bc=[]
brc=[]
for i in range(7):
    d=regr1.intercept_*pow(l[i],-1)
    bc.append(d)
print(bc)
for j in range(7):
    e=regr1.coef_[0]*pow(l[j],-AAE)
    brc.append(e)
print(brc)
total=[]
for m in range(7):
 f=bc[m]+brc[m]
 total.append(f)
print(total)


# In[531]:


plt.scatter(l, total, color = 'blue')
plt.plot(l, total, color = 'blue')
plt.scatter(l, bc, color = 'black')
plt.plot(l, bc, color = 'black')
plt.scatter(l, brc, color = 'brown')
plt.plot(l, brc, color = 'brown')
plt.title('ATN vs wavelength')
plt.xlabel('wavelenth (nm)')
plt.ylabel('ATN (Mm-1)')
plt.legend(["Total","BC","BrC"])
plt.show()
aa= total[1]/total[5]
bb= 470/880
aaa= math.log(aa)
bbb= math.log(bb)
#print(aaa)
AAE_470_880 = -(aaa/bbb)
#print(total[1],total[5])
print("AAE470/880 = ",AAE_470_880 )

cc= brc[1]/brc[4]
dd= 370/660
ccc= math.log(cc)
ddd= math.log(dd)
#print(aaa)
AAE_brcc_370_660 = -(ccc/ddd)
#print(total[1],total[5])
print("AAEbrc = ",AAE_brcc_370_660 )


# In[532]:


#percentage contributions
perc_bc=[]
perc_brc=[]
for i in range (7):
 o = (bc[i]/total[i])*100
 perc_bc.append(o)
for j in range (7):
 q =  (brc[j]/total[j])*100
 perc_brc.append(q)
print(perc_bc)
print(perc_brc)


# In[533]:


plt.bar(l, perc_brc, color='brown', width=40)
plt.bar(l, perc_bc, bottom=perc_brc, color='black', width=40)
plt.xlabel('wavelenth')
plt.ylabel('Percentage of Absorption')
plt.legend(["BrC","BC"],bbox_to_anchor = (1.05, 0.6))
plt.show()
print(l)


# In[534]:


abs_bc=[]
abs_brc=[]

for i in range(7):   
 for j in range(197504,198139):
    print(i,j)
    print(abs[i][j])
    z=abs[i][j]*(perc_bc[i]/100)
    abs_bc.append(z)
    u=abs[i][j]*(perc_brc[i]/100)
    abs_brc.append(u)
    
abs_bc= np.array(abs_bc)
abs_bc=abs_bc.reshape(7,635)
abs_bc= np.transpose(abs_bc)
abs_brc=np.array(abs_brc)
abs_brc=abs_brc.reshape(7,635)
abs_brc= np.transpose(abs_brc)
print(abs_bc[2][4])
print(abs)
print(abs_bc,"\n\n\n\n\n")
print(abs_brc)
print(np.shape(abs_bc))
#abs=np.transpose(abs)
#np.shape(abs)
#for i in range (1907,10612):
    #for j in range (7):
       # print(abs[i][j])


# In[16]:


print(abs_bc[:,0])
BC_CONC=[]
ABS=[]
OC=[]
bc_conc = [abs_bc[:,0]/18.47,abs_bc[:,1]/14.54 , abs_bc[:,2]/13.14, abs_bc[:,3]/11.58 , abs_bc[:,4]/10.35, abs_bc[:,5]/7.77, abs_bc[:,6]/7.19]
print(bc_conc)
bc_conc=np.array(bc_conc)
#bc_conc=bc_conc.reshape(7,200)
print(bc_conc[:,0])
print(bc_conc)
bc_conc=np.transpose(bc_conc)
print(bc_conc)
for i in range (720):
 for j in range (7):
    ll=bc_conc[i][j]
    BC_CONC.append(ll)
BC_CONC=np.array(BC_CONC)
BC_CONC=BC_CONC.reshape(720,7)
print(BC_CONC)
for i in range (1907,10612):
   for j in range (7): 
     vv=abs[i][j]
     ABS.append(vv)
ABS=np.array(ABS)
ABS = ABS.reshape(8705,7)
#ABS = np.transpose(ABS)
print(ABS[0])
for i in range (1907,10612):
   for j in range (7): 
     mm=oc[j][i]
     OC.append(mm)
OC=np.array(OC)
OC = OC.reshape(8705,7)
#ABS = np.transpose(ABS)
print(OC[0])


# In[96]:



abs=np.transpose(abs)
abs_pri=[]
abs_brc_sec=[]
abs_brc_sec_1=[]
for i in range(150):
 for j in range (720):
  for k in range (7):
    s=(ABS[j][k]-i*bc_conc[j][k])
    if(s>0):
     abs_brc_sec.append(s)
    else:
     abs_brc_sec.append(0)

print (abs_brc_sec)
np.shape(abs) 
np.shape(abs_brc_sec)


# In[183]:


arr=[]
abs_brc_sec_=np.array(abs_brc_sec_)
arr=abs_brc_sec_.reshape(100,720,7)
print(arr)
print(arr[0][:,6])
print(arr[0].shape)
print(abs_bc.shape)
print(abs_bc)
print(abs_bc[:,0])
#for i in range (100):
 #for j in range (200):
 # for k in range (7):
  ## print(arr[i][j][k],end=' ')
 #  print("\n")


# In[184]:


from sklearn.linear_model import LinearRegression
regress = LinearRegression()
from sklearn.metrics import r2_score
R2=[]
BC_CONC_pred=[]
B= [1 for i in range (720)]
#print (B)
for i in range(100):
  for j in range(7):
    T = np.concatenate([arr[i][:,j],B])
    T=T.reshape(-1,1)
    T=np.transpose(T)
    T = T.reshape(2,720)
    T = np.transpose(T)
    #print(T)
    regress.fit(T,BC_CONC[:,j])
    BC_CONC_pred.append(regress.predict(T))
BC_CONC_pred=np.array(BC_CONC_pred)
BC_CONC_pred=BC_CONC_pred.reshape(100,720,7)
for i in range(100):
  for j in range(7):
    r2=r2_score(BC_CONC[:,j],BC_CONC_pred[i][:,j])
    R2.append(r2)
#print(R2)
R2=np.array(R2)
R2_arr=R2.reshape(100,7)
print(R2_arr)
print(R2_arr[:,0])


# In[186]:


a_list = list( np.arange(0.0, 10.0, 0.1))
plt.scatter(a_list,  R2_arr[:,0], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,0] , color = 'blue')


# In[187]:


plt.scatter(a_list,  R2_arr[:,1], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,1] , color = 'blue')


# In[188]:


plt.scatter(a_list,  R2_arr[:,2], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,2] , color = 'blue')


# In[189]:


plt.scatter(a_list,  R2_arr[:,3], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,3] , color = 'blue')


# In[190]:


plt.scatter(a_list,  R2_arr[:,4], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,4] , color = 'blue')


# In[191]:


plt.scatter(a_list,  R2_arr[:,5], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,5] , color = 'blue')


# In[192]:


plt.scatter(a_list,  R2_arr[:,6], color = 'blue')
#plt.ylim(-1,1)
plt.plot(a_list,  R2_arr[:,6] , color = 'blue')


# In[170]:


abs_brc_sec_370=[]
for i in range (720):
    #print(i, abs[i][0], abs[0][i],BC_CONC[i][0]) 
    q=((OC[i][0]/1000)-1.9*BC_CONC[i][0])
    if(q>0):
     abs_brc_sec_370=np.append(abs_brc_sec_370,q)
    else:
     abs_brc_sec_370=np.append(abs_brc_sec_370,0)
print(abs_brc_sec_370)
abs_brc_sec_370=np.array(abs_brc_sec_370)
arr_370=abs_brc_sec_370.reshape(720,1)
print(arr_370)
print(arr_370[0])


# In[363]:


print(abs_brc)
print(abs_brc[:,0])


# In[149]:


perc_sec=[]
for i in range (720):
    print(abs_brc[i][0])
    print(arr_370[i])
    frac=(arr_370[i]/abs_brc[i][0])*100
    perc_sec.append(frac)
perc_sec=np.array(perc_sec)
print(perc_sec)
perc_370=frac
print(frac)


# In[129]:


abs_brc_sec_470=[]
for j in range (200):
    #t= i*abs_bc[k][j]
    #abs_pri.append(t)
    #print(abs[j][k])
    #print(abs_bc[j][k])
    q=(abs[j][1]-18*bc_conc[j][1])
    if(q>0):
     abs_brc_sec_470=np.append(abs_brc_sec_470,q)
    else:
     abs_brc_sec_470=np.append(abs_brc_sec_470,0)
print(abs_brc_sec_470)
abs_brc_sec_470=np.array(abs_brc_sec_470)
arr_470=abs_brc_sec_470.reshape(200,1)
print(arr_470)
print(arr_470[:,0])


# In[136]:


perc_sec_470=[]
for i in range (200):
    frac=(arr_470[i]/abs_brc[i][1])*100
    perc_sec_470.append(frac)
print(perc_sec_470)
perc_470=frac
print(perc_470)


# In[82]:


abs_brc_sec_520=[]
for j in range (168):
    q=(abs[2][j]-146*bc_conc[j][2])
    if(q>0):
     abs_brc_sec_520=np.append(abs_brc_sec_520,q)
    else:
     abs_brc_sec_520=np.append(abs_brc_sec_520,0)
print(abs_brc_sec_520)
abs_brc_sec_520=np.array(abs_brc_sec_520)
arr_520=abs_brc_sec_520.reshape(168,1)
print(arr_520)
print(arr_520[:,0])


# In[86]:


perc_sec_520=[]
for i in range (168):
    frac=(arr_520[i]/abs_brc[i][2])*100
    perc_sec_520.append(frac)
print(perc_sec_520)
perc_520=frac
print(perc_520)


# In[143]:


abs_brc_sec_590=[]
for j in range (200):
    q=(abs[j][3]-9*bc_conc[j][3])
    if(q>0):
     abs_brc_sec_590=np.append(abs_brc_sec_590,q)
    else:
     abs_brc_sec_590=np.append(abs_brc_sec_590,0)
print(abs_brc_sec_590)
abs_brc_sec_590=np.array(abs_brc_sec_590)
arr_590=abs_brc_sec_590.reshape(200,1)
print(arr_590)
print(arr_590[:,0])


# In[144]:


perc_sec_590=[]
for i in range (200):
    frac=(arr_590[i]/abs_brc[i][3])*100
    perc_sec_590.append(frac)
print(perc_sec_590)
perc_590=frac
print(perc_590)


# In[182]:


abs_brc_sec_=[]
for i in np.arange(0.0, 10.0, 0.1):
 for j in range (720):
  for k in range (7):
    s=((OC[j][k]/1000)-i*(bc_conc[j][k]))
    if(s>0):
     abs_brc_sec_.append(s)
    else:
     abs_brc_sec_.append(0)

print (abs_brc_sec_)
np.shape(abs) 
np.shape(abs_brc_sec_)


# In[387]:


#abs_bc = np.transpose(abs_bc)
print(np.shape(abs_bc))
abs_bc_daily=[]
sum=0
for i in range (7):
 for j in range (8705):
    #print(j)
    #print(abs_bc[j][i])
    count=j
    if (count%24==0 and count!=0):
        #j+=1
        abs_bc_daily.append((sum/24))
        sum=0
    else:
     sum+=abs_bc[j][i]
     #print(sum)
#print(abs_bc_daily)
#print(np.shape(abs_bc_daily))
abs_bc_daily=np.array(abs_bc_daily)
abs_bc_daily=abs_bc_daily.reshape(7,362)
#print(abs_bc_daily)
abs_bc_daily = np.transpose(abs_bc_daily)
print(np.shape(abs_bc_daily))
print(abs_bc_daily)
#print(abs_bc_daily[:,0])


# In[388]:


#jan-mar daily bc 
abs_bc_quarter1_daily_2016=[]
for i in range (91):
 for j in range (7):
  uu=abs_bc_daily[i][j]
  abs_bc_quarter1_daily_2016.append(uu)
    
abs_bc_quarter1_daily_2016=np.array(abs_bc_quarter1_daily_2016)
abs_bc_quarter1_daily_2016 =abs_bc_quarter1_daily_2016.reshape(91,7)
a_list = list(range(1,92))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.plot(a_list,  abs_bc_quarter1_daily_2016[:,0] , color = 'black')


# In[38]:



abs_bc_quarter1_2016=[]
for i in range (2184):
 for j in range (7):
  uu=abs_bc[i][j]
  abs_bc_quarter1_2016.append(uu)
    
abs_bc_quarter1_2016=np.array(abs_bc_quarter1_2016)
abs_bc_quarter1_2016 =abs_bc_quarter1_2016.reshape(2184,7)
#print(abs_bc_quarter1_2016)
a_list = list(range(1,2185))
#print(a_list)
plt.scatter(a_list,  abs_bc_quarter1_2016[:,0], color = 'black')
#plt.ylim(-1,1)
f = plt.figure()
f.set_figwidth(100)
f.set_figheight(50)
plt.plot(a_list,  abs_bc_quarter1_2016[:,0] , color = 'black')


# In[389]:


#apr-jun daily bc 
abs_bc_quarter2_daily_2016=[]
for i in range (91,182):
 for j in range (7):
  uu=abs_bc_daily[i][j]
  abs_bc_quarter2_daily_2016.append(uu)
    
abs_bc_quarter2_daily_2016=np.array(abs_bc_quarter2_daily_2016)
abs_bc_quarter2_daily_2016 =abs_bc_quarter2_daily_2016.reshape(91,7)
a_list = list(range(1,92))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.ylim(0,100)
plt.plot(a_list,  abs_bc_quarter2_daily_2016[:,0] , color = 'black')


# In[390]:


#jul-sept daily bc 
abs_bc_quarter3_daily_2016=[]
for i in range (182,274):
 for j in range (7):
  uu=abs_bc_daily[i][j]
  abs_bc_quarter3_daily_2016.append(uu)
    
abs_bc_quarter3_daily_2016=np.array(abs_bc_quarter3_daily_2016)
abs_bc_quarter3_daily_2016 =abs_bc_quarter3_daily_2016.reshape(92,7)
a_list = list(range(1,93))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.ylim(0,100)
plt.plot(a_list,  abs_bc_quarter3_daily_2016[:,0] , color = 'black')


# In[392]:


#oct-dec daily bc 
abs_bc_quarter4_daily_2016=[]
for i in range (274,362):
 for j in range (7):
  uu=abs_bc_daily[i][j]
  abs_bc_quarter4_daily_2016.append(uu)
    
abs_bc_quarter4_daily_2016=np.array(abs_bc_quarter4_daily_2016)
abs_bc_quarter4_daily_2016 =abs_bc_quarter4_daily_2016.reshape(88,7)
a_list = list(range(1,89))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
#plt.ylim(0,100)
plt.plot(a_list,  abs_bc_quarter4_daily_2016[:,0] , color = 'black')


# In[393]:


print(np.shape(abs_brc))
abs_brc_daily=[]
sum=0
for i in range (7):
 for j in range (8705):
    print(j)
    print(abs_brc[j][i])
    count=j
    if (count%24==0 and count!=0):
        #j+=1
        abs_brc_daily.append((sum/24))
        sum=0
    else:
     sum+=abs_brc[j][i]
     print(sum)
#print(abs_bc_daily)
#print(np.shape(abs_bc_daily))
abs_brc_daily=np.array(abs_brc_daily)
abs_brc_daily=abs_brc_daily.reshape(7,362)
#print(abs_bc_daily)
abs_brc_daily = np.transpose(abs_brc_daily)
print(np.shape(abs_brc_daily))
print(abs_brc_daily)
#print(abs_bc_daily[:,0])


# In[394]:


#jan-mar daily brc 
abs_brc_quarter1_daily_2016=[]
for i in range (91):
 for j in range (7):
  uu=abs_brc_daily[i][j]
  abs_brc_quarter1_daily_2016.append(uu)
    
c)
a_list = list(range(1,92))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.plot(a_list,  abs_brc_quarter1_daily_2016[:,0] , color = 'brown')


# In[399]:


#apr-jun daily brc 
abs_brc_quarter2_daily_2016=[]
for i in range (91,182):
 for j in range (7):
  uu=abs_brc_daily[i][j]
  abs_brc_quarter2_daily_2016.append(uu)
    
abs_brc_quarter2_daily_2016=np.array(abs_brc_quarter2_daily_2016)
abs_brc_quarter2_daily_2016 =abs_brc_quarter2_daily_2016.reshape(91,7)
a_list = list(range(1,92))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.ylim(0,50)
plt.plot(a_list,  abs_brc_quarter2_daily_2016[:,0] , color = 'brown')


# In[396]:


#jul-sept daily brc 
abs_brc_quarter3_daily_2016=[]
for i in range (182,274):
 for j in range (7):
  uu=abs_brc_daily[i][j]
  abs_brc_quarter3_daily_2016.append(uu)
    
abs_brc_quarter3_daily_2016=np.array(abs_brc_quarter3_daily_2016)
abs_brc_quarter3_daily_2016 =abs_brc_quarter3_daily_2016.reshape(92,7)
a_list = list(range(1,93))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
#plt.ylim(0,40)
plt.plot(a_list,  abs_brc_quarter3_daily_2016[:,0] , color = 'brown')


# In[397]:


#oct-dec daily brc 
abs_brc_quarter4_daily_2016=[]
for i in range (274,362):
 for j in range (7):
  uu=abs_brc_daily[i][j]
  abs_brc_quarter4_daily_2016.append(uu)
    
abs_brc_quarter4_daily_2016=np.array(abs_brc_quarter4_daily_2016)
abs_brc_quarter4_daily_2016 =abs_brc_quarter4_daily_2016.reshape(88,7)
a_list = list(range(1,89))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
#plt.ylim(0,20)
plt.plot(a_list,  abs_brc_quarter4_daily_2016[:,0] , color = 'brown')


# In[136]:


print(np.shape(abs_brc))
abs_daily=[]
sum=0
for i in range (7):
 for j in range (1907,10611):
    print(j)
    print(abs[j][i])
    count=j
    if (count%24==0 and count!=0):
        #j+=1
        abs_daily.append((sum/24))
        sum=0
    else:
     sum+=abs[j][i]
     print(sum)
#print(abs_bc_daily)
#print(np.shape(abs_bc_daily))
abs_daily=np.array(abs_daily)
abs_daily=abs_daily.reshape(7,363)
#print(abs_bc_daily)
abs_daily = np.transpose(abs_daily)
print(np.shape(abs_daily))
print(abs_daily)
#print(abs_bc_daily[:,0])


# In[137]:


#jan-mar daily total
abs_quarter1_daily_2016=[]
for i in range (91):
 for j in range (7):
  uu=abs_daily[i][j]
  abs_quarter1_daily_2016.append(uu)
    
abs_quarter1_daily_2016=np.array(abs_quarter1_daily_2016)
abs_quarter1_daily_2016 =abs_quarter1_daily_2016.reshape(91,7)
a_list = list(range(1,92))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.plot(a_list,  abs_quarter1_daily_2016[:,0] , color = 'green')


# In[143]:


#apr-jun daily brc 
abs_quarter2_daily_2016=[]
for i in range (91,182):
 for j in range (7):
  uu=abs_daily[i][j]
  abs_quarter2_daily_2016.append(uu)
    
abs_quarter2_daily_2016=np.array(abs_quarter2_daily_2016)
abs_quarter2_daily_2016 =abs_quarter2_daily_2016.reshape(91,7)
a_list = list(range(1,92))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
plt.ylim(0,200)
plt.plot(a_list,  abs_quarter2_daily_2016[:,0] , color = 'green')


# In[144]:


#jul-sept daily bc 
abs_quarter3_daily_2016=[]
for i in range (182,274):
 for j in range (7):
  uu=abs_daily[i][j]
  abs_quarter3_daily_2016.append(uu)
    
abs_quarter3_daily_2016=np.array(abs_quarter3_daily_2016)
abs_quarter3_daily_2016 =abs_quarter3_daily_2016.reshape(92,7)
a_list = list(range(1,93))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
#plt.ylim(0,40)
plt.plot(a_list,  abs_quarter3_daily_2016[:,0] , color = 'green')


# In[145]:


#oct-dec daily bc 
abs_quarter4_daily_2016=[]
for i in range (274,362):
 for j in range (7):
  uu=abs_daily[i][j]
  abs_quarter4_daily_2016.append(uu)
    
abs_quarter4_daily_2016=np.array(abs_quarter4_daily_2016)
abs_quarter4_daily_2016 =abs_quarter4_daily_2016.reshape(88,7)
a_list = list(range(1,89))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
plt.xlim(0,90)
#plt.ylim(0,20)
plt.plot(a_list,  abs_quarter4_daily_2016[:,0] , color = 'green')


# In[194]:


math.log(5)


# In[298]:


tot=0
AAE_monthly_2016=[1.481,1.348,1.356,1.364,1.269,1.331,1.222,1.353,1.166,1.759,1.347,1.526]
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
#print(AAE_monthly_2016)
#a_list = list(range(1,13))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
#plt.xlim(0,90)
#plt.ylim(0,20)
plt.ylabel('AAE 470/880')
plt.plot(months,  AAE_monthly_2016 , color = 'purple')
for i in range (12):
 tot+=AAE_monthly_2016[i]
avg_AAE_2016=tot/12
print(avg_AAE_2016)


# In[325]:


brown_470_2016=[]
black_470_2016=[62.61,75.22,78.22,63.39,84.05,80.59,86.49,78.12,89.92,48.27,78.32,61.60]
for i in range(12):
 m=100-black_470_2016[i]
 brown_470_2016.append(m)
print(brown_470_2016)
x = np.arange(12)
width=0.5
plt.bar(x-0.2, black_470_2016, width, color='black')
plt.bar(x+0.2, brown_470_2016, width, color='brown')
plt.xticks(x, months)
plt.ylim(0,100)
plt.ylabel("Percentage of Abs at 470 nm ")
plt.legend(["BC", "Brc"],bbox_to_anchor = (1.05, 0.6))
plt.show()


# In[324]:


brown_370_2016=[]
black_370_2016=[51.53,62.54,57.99,55.93,56.82,53.28,64.30,58.99,64.80,36.10,59.85,49.26]
for i in range(12):
 m=100-black_370_2016[i]
 brown_370_2016.append(m)
print(brown_370_2016)
x = np.arange(12)
width=0.45
plt.bar(x-0.2, black_370_2016,width, color='black')
plt.bar(x+0.2, brown_370_2016,width, color='brown')
plt.xticks(x, months)
plt.ylim(0,100)
plt.ylabel("Percentage of Abs at 370 nm ")
plt.legend(["BC", "Brc"],bbox_to_anchor = (1.05, 0.6))
plt.show()


# In[402]:


ss=0
for j in range (1907,10612):
  print(j,abs[j][0])
  ss+=abs[j][0]
print(ss/8705)


# In[479]:


tot=0
AAE_BrC_monthly_2016=[1.701,2.053,2.933,1.364,3.989,3.754,3.696,2.816,4.46,1.819,2.758,1.818]
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
#print(AAE_monthly_2016)
#a_list = list(range(1,13))
f = plt.figure()
f.set_figwidth(15)
f.set_figheight(3.23)
#plt.xlim(0,90)
#plt.ylim(0,20)
plt.ylabel('AAE BrC 370/660') 
plt.plot(months,  AAE_BrC_monthly_2016 , color = 'orange')
for i in range (12):
 tot+=AAE_BrC_monthly_2016[i]
avg_AAE_2016=tot/12
print(avg_AAE_2016)


# In[38]:


n
ABS=[]
for i in range (7):
  for j in range (1913,10612): 
    vv=abs[i][j]
    ABS.append(vv)
ABS=np.array(ABS)
ABS = ABS.reshape(7,8699)
#ABS=np.transpose(ABS)
abs_day_2016=[]
abs_night_2016=[]
for i in range (7):
 for j in range (8688):
   if (j%12==0 and (j/12)%2==0):
       count=j
       n = count+12
       for k in range (count,n):
         print(i,k,j,ABS[i][k])
         h=ABS[i][k]
         abs_day_2016.append(h)
       j+=1
   if (j%12==0 and (j/12)%2!=0):
       counter=j
       for m in range (counter,(counter+12)):
         print(i,m,j,ABS[i][m],"nn")
         g=ABS[i][m]
         abs_night_2016.append(g)
       j+=1
abs_day_2016=np.array(abs_day_2016)
abs_day_2016 =abs_day_2016.reshape(7,4344) 
abs_day_2016 = np.transpose(abs_day_2016)

abs_night_2016=np.array(abs_night_2016)
abs_night_2016 =abs_night_2016.reshape(7,4344) 
abs_night_2016 = np.transpose(abs_night_2016)
print(abs_day_2016)
print(abs_night_2016)
print(np.shape(abs_day_2016))


# In[260]:


black_day_2016=[64.75,72.45,77.03,61.9,69.47,73.85,80.8,73.69,83.091,45.97,74.05,61.65]
black_night_2016=[69.62,72.59,73.51,74.54,77.10,79.85,66.413,81,83.46,76.87,73.55,59.88]
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
x = np.arange(12)
width=0.4
plt.bar(x-0.2, black_day_2016, width, color='orange')
plt.bar(x+0.2, black_night_2016, width, color='green')
plt.xticks(x, months)
plt.ylim(0,100)
plt.ylabel("Average Percentage of Abs of BC  ")
plt.legend(["Day", "Night"],bbox_to_anchor = (1.05, 0.6))
plt.show()


# In[266]:


brown_day_2016=[]
brown_night_2016=[]
for i in range(12):
 m=100-black_day_2016[i]
 brown_day_2016.append(m)
 n=100-black_night_2016[i]
 brown_night_2016.append(n)
print(brown_day_2016)
print(brown_night_2016)
months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sept','Oct','Nov','Dec']
x = np.arange(12)
width=0.4
plt.bar(x-0.2, brown_day_2016, width, color='skyblue')
plt.bar(x+0.2, brown_night_2016, width, color='indigo')
plt.xticks(x, months)
plt.ylim(0,100)
plt.ylabel("Average Percentage of Abs of BrC  ")
plt.legend(["Day", "Night"],bbox_to_anchor = (1.3, 0.8))
plt.show()


# In[535]:


#year average 
abs_bc_yearly_2016=[]
abs_brc_yearly_2016=[]
for i in range(7):  
    z=sum(abs_bc[:,i]/len(abs_bc[:,i]))
    abs_bc_yearly_2016.append(z)
abs_bc_yearly_2016= np.array(abs_bc_yearly_2016)
print(abs_bc_yearly_2016)
for i in range(7):  
    y=sum(abs_brc[:,i]/len(abs_brc[:,i]))
    abs_brc_yearly_2016.append(y)
abs_brc_yearly_2016= np.array(abs_brc_yearly_2016)
print(abs_brc_yearly_2016)
for i in range(7):  
    ll=sum(abs_bc_yearly_2016/len(abs_bc_yearly_2016))
    abs_bc_yearly_2016_average=ll
for i in range(7):  
    kk=sum(abs_brc_yearly_2016)/7
    abs_brc_yearly_2016_average=kk
print("Avg BC_2016=" , abs_bc_yearly_2016_average)
print("Avg BrC_2016=", abs_brc_yearly_2016_average)


# In[573]:


lat=[28.58,30.73,25.3,18.53,17.71,21.10,26.30,08.48,23.25,11.66,30.25]
long=[77.20,76.88,83.01,73.85,83.23,79.05,73.01,76.95,69.66,92.71,78.08]
colours=[116.29,92.01,72.38,72.11,62.14,45.53,38.47,33.79,22.7,19.30,13.84]
text = ["NDL", "CHD", "Varanasi", "Pune", "Vizag",
        "Nagpur", "Jodhpur", "Trivandrum", "Bhuj", "Port Blair","Ranichuri"]
f = plt.figure()
f.set_figwidth(13)
f.set_figheight(9)
plt.scatter(long,lat,s=300,c=colours)
plt.colorbar(label="Absorption annual avg Mm-1", shrink=0.6)
plt.title("Avg BC absorptions 2016" )
plt.xlabel("Long")
plt.ylabel("Lat")
plt.ylim(5,35)
plt.xlim(65,100)
for i in range(len(lat)):
    plt.annotate(text[i], (long[i]+0.8, lat[i]+0.3))
plt.show()


# In[188]:


lat=[28.58,30.73,25.3,18.53,17.71,21.10,26.30,08.48,23.25,11.66,30.25]
long=[77.20,76.88,83.01,73.85,83.23,79.05,73.01,76.95,69.66,92.71,78.08]
colours=[86.26,37.06,22.93,24.53,18.37,21.54,40.64,24.63,12.21,13.02,18.35]
text = ["NDL", "CHD", "Varanasi", "Pune", "Vizag",
        "Nagpur", "Jodhpur", "Trivandrum", "Bhuj", "Port Blair","Ranichuri"]
f = plt.figure()
f.set_figwidth(13)
f.set_figheight(9)
plt.scatter(long,lat,s=300,c=colours)
plt.colorbar(label="Absorption annual avg Mm-1", shrink=0.6)
plt.title("Avg BrC absorptions 2016" )
plt.xlabel("Long")
plt.ylabel("Lat")
plt.ylim(5,35)
plt.xlim(65,100)
for i in range(len(lat)):
    plt.annotate(text[i], (long[i]+0.8, lat[i]+0.3))
plt.show()


# In[227]:


A=[1,2,3,4,5,6,7,8,9,10,11,12,13,14]
for i in range (1,6)+ range(9,12):
 print(A[i])


# In[ ]:




