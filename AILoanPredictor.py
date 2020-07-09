import warnings
warnings.filterwarnings("ignore", "(?s).*MATPLOTLIBDATA.*", category=UserWarning)
from tkinter import filedialog ,messagebox
import tkinter
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from tkinter import *
from tkinter import filedialog
from PIL import ImageTk,Image
from csv import writer
from sklearn.ensemble import BaggingClassifier

Xnew = pd.DataFrame()
Ynew = pd.DataFrame()


#--------main-----------#
def main():
    root=Tk()
    obj=Start1(root)
    #obj=PG3(root,"train_dataset.csv","test_dataset.csv")
    root.mainloop()
#-----------------------#



#-----------------------------------------------------Start1---------------------------------------------------#
class Start1:		  
    def __init__(self, root):
        self.root = root
        self.root.title("ARTIFICIAL INTELLIGENCE")
        self.root.geometry("1350x700+0+0")
        self.im =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 1.jpg") 
        self.im1 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 0.png")
        self.Username = StringVar()
        self.Password = StringVar()
        img1=Label(self.root,image=self.im).pack()
        frame =Frame(self.root,bg="#203a51")
        frame.place(x=500,y=200,height = 425,width = 400)
        img2=Label(frame,image=self.im1,bd=0).grid(row=0,columnspan=2,pady=15)
        txt1=Label(frame,text="Username",compound=LEFT,font=("times new roman",20,"bold"),bg="#22394b",fg = "WHITE").grid(row=1,column=0,padx=0,pady=3)
        txt2=Entry(frame,bd=5,textvariable=self.Username,relief=GROOVE,font=("",15)).grid(row=1,column=1,padx=20,pady=3)
        txt3=Label(frame,text="Password",compound=LEFT,font=("times new roman",20,"bold"),bg="#22394b", fg = "WHITE").grid(row=2,column=0,padx=0,pady=3)
        txt4=Entry(frame,bd=5,textvariable=self.Password,relief=GROOVE,font=("",15)).grid(row=2,column=1,padx=20,pady=3)
        txt5=Button(frame,text="Login",width = 15,font=("times new roman",20,"bold"),bg = "#22394b",fg = "WHITE", command =self.Login).grid(row =3 ,column = 1 ,pady = 10)
    #-----------------------------------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------------------------------#
    def Login(self):  	
        if self.Username.get()== "" or self.Password.get()== "" :
            messagebox.showerror("Error","All Fields Are Required") 
        elif self.Username.get()== "AI" and self.Password.get()== "12345678":
            messagebox.showinfo("Valid","Login Successful")
            command = self.NewWin()
        else:
            messagebox.showerror("Error","Login failed")
    #----------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------#
    def NewWin(self):
        self.top = Toplevel(self.root)
        self.obj = PG1(self.top)      
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#




#----------------------------------------------NEW PAGE--------------------------------------------------#
class PG1:         
    def __init__(self, root):
        self.root = root
        self.root.title("LOAN PREDICTOR")
        self.root.geometry("1350x700+0+0")
        self.filename = "No File Selected"
        self.testfile=""
        self.trainfile=""
        self.im =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 1.jpg") 
        self.im1 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 2.png")
        img1=Label(self.root,image=self.im).pack()
        frame =Frame(self.root,bg="#203a51")
        frame.place(x=395,y=0,height = 800,width = 530)
        img2=Label(frame,image=self.im1,bd=0).grid(row=0,columnspan=5,pady=15)
        txt1=Label(frame,text="Input_Files: ",compound=LEFT,font=("times new roman",15),bg="#22394b",fg = "WHITE").grid(row=1,column=2,padx=0,pady=1)
        txt2=Button(frame,text="Train_Data",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Load_train).grid(row =2 ,column = 1,pady = 10)
        txt3=Button(frame,text="Test_Data",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Load_test).grid(row =2 ,column = 3 ,pady = 10)
        
        txt4=Label(frame,text="Histogram: ",compound=LEFT,font=("times new roman",15),bg="#22394b",fg = "WHITE").grid(row=3,column=2,padx=0,pady=1)
        txt5=Button(frame,text="Coapplicant_Income",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command=self.CoapplicantIncome).grid(row =4 ,column = 1 ,pady = 10)
        txt6=Button(frame,text="Loan_Amount",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command=self.LoanAmount).grid(row =4 ,column = 2 ,pady = 10)
        txt7=Button(frame,text="Loan_Amount_Term",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Loan_Amount_Term).grid(row =4 ,column = 3,pady = 10)
        txt8=Button(frame,text="Credit_History",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Credit_History).grid(row =5 ,column = 1 ,pady = 10)
        txt10=Button(frame,text="ApplicantIncome",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Applicant_Income).grid(row =5 ,column = 2 ,pady = 10)
        
        txt4=Label(frame,text="Box Plot: ",compound=LEFT,font=("times new roman",15),bg="#22394b",fg = "WHITE").grid(row=6,column=2,padx=0,pady=1)
        txt9=Button(frame,text="Applicant_Income",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.ApplicantIncome).grid(row =7 ,column = 1 ,pady = 10)
        txt10=Button(frame,text="Loan_Amount",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.LoanAmount1).grid(row =7 ,column = 2 ,pady = 10)
        
        txt11=Label(frame,text="Bar Graph: ",compound=LEFT,font=("times new roman",15),bg="#22394b",fg = "WHITE").grid(row=8,column=2,padx=0,pady=1)
        txt12=Button(frame,text="Credit_History",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command=self.CreditHistory).grid(row =9 ,column = 1 ,pady = 10)
        txt13=Button(frame,text="Gender",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command=self.Gender).grid(row =9 ,column = 2 ,pady = 10)
        txt14=Button(frame,text="Dependents",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command=self.Dependents).grid(row =9 ,column = 3 ,pady = 10)
        txt15=Button(frame,text="Self_Employed",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Self_Employed).grid(row =10 ,column = 1,pady = 3)
        txt16=Button(frame,text="Married",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command =self.Married).grid(row =10 ,column = 2 ,pady = 3)
        
        txt17=Button(frame,text="Data Preparation",width = 15,font=("times new roman",15),bg = "#1e6ead",fg = "WHITE",command =self.Prep).grid(row =11 ,column = 2 ,pady = 1)
    
    #--------------------------------------------------------------------------------------------------------#    
    #--------------------------------------------------------------------------------------------------------#

    def Load_train(self):
        self.trainfile = filedialog.askopenfilename(initialdir = "/"
            ,title = "Browse File" , filetypes = (("Excel Files",".csv"),("all files","*.*")))
        if self.trainfile != "":
            messagebox.showinfo("File Open Successful","File Directory : " + self.trainfile) 
        else:
            messagebox.showerror("Error","File Input Failed")

    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Load_test(self):
        self.testfile = filedialog.askopenfilename(initialdir = "/"
            ,title = "Browse File" , filetypes = (("Excel Files",".csv"),("all files","*.*")))
        if self.testfile != "":
            messagebox.showinfo("File Open Successful","File Directory : " + self.testfile)   
        else:
            messagebox.showerror("Error","File Input Failed")
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Applicant_Income(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train['ApplicantIncome'].hist(bins=50,color = "#EEA47FFF")
            plt.xlabel('ApplicantIncome',color = "#EEA47FFF")
            plt.tick_params('x',colors='#00539CFF')
            plt.tick_params('y',colors='#00539CFF')
            plt.show()   
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")        
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def CoapplicantIncome(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train['CoapplicantIncome'].hist(bins=50,color ='#DF6589FF')
            plt.xlabel('CoapplicantIncome',color = '#DF6589FF')
            plt.tick_params('x',colors="#3C1053FF")
            plt.tick_params('y',colors="#3C1053FF")
            plt.show()   
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def LoanAmount(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train['LoanAmount'].hist(bins=50,color = "#E94B3CFF")
            plt.xlabel('Loanamount',color = "#E94B3CFF")
            plt.tick_params('x',colors='#2D2926FF')
            plt.tick_params('y',colors='#2D2926FF')
            plt.show()   
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")            
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Loan_Amount_Term(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train['Loan_Amount_Term'].hist(bins=50,color = "#D64161FF")
            plt.xlabel('Looan _Amount_Term',color = "#D64161FF")
            plt.tick_params('x',colors='#435E55FF')
            plt.tick_params('y',colors='#435E55FF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")    
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Credit_History(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train['Credit_History'].hist(bins=50,color = "#4B878BFF")
            plt.xlabel('Credit_History',color = "#4B878BFF")
            plt.tick_params('x',colors='#D01C1FFF')
            plt.tick_params('y',colors='#D01C1FFF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def ApplicantIncome(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train.boxplot(column='ApplicantIncome')
            plt.xlabel('ApplicantIncome')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#                         

    def LoanAmount1(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            train.boxplot(column='LoanAmount')
            plt.xlabel('LoanAmount')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")   




    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#




    def CreditHistory(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            temp3=pd.crosstab(train['Credit_History'],train['Loan_Status'])
            temp3.plot(kind='barh',stacked=True,color=['#184A45FF','#CE4A7EFF'],grid=False)
            plt.tick_params('x',colors='#CE4A7EFF')
            plt.tick_params('y',colors='#CE4A7EFF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")      
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Gender(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            temp3=pd.crosstab(train['Gender'],train['Loan_Status'])
            temp3.plot(kind='barh',stacked=True,color=['#00B1D2FF','#FFD662FF'],grid=False)
            plt.tick_params('x',colors='#00B1D2FF')
            plt.tick_params('y',colors='#00B1D2FF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")  
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Dependents(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            temp3=pd.crosstab(train['Dependents'],train['Loan_Status'])
            temp3.plot(kind='barh',stacked=True,color=['#00539CFF','#FFD662FF'],grid=False)
            plt.tick_params('x',colors='#D64161FF')
            plt.tick_params('y',colors='#D64161FF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#
    
    def Self_Employed(self):

        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            temp3=pd.crosstab(train['Self_Employed'],train['Loan_Status'])
            temp3.plot(kind='barh',stacked=True,color=['#184A45FF','#FC766AFF'],grid=False)
            plt.tick_params('x',colors='#FC766AFF')
            plt.tick_params('y',colors='#FC766AFF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Married(self):
        if self.testfile != "" and self.trainfile != "" :
            train = pd.read_csv(self.trainfile)
            temp3=pd.crosstab(train['Married'],train['Loan_Status'])
            temp3.plot(kind='barh',stacked=True,color=['#5F4B8BFF','#E69A8DFF'],grid=False)
            plt.tick_params('x',colors='#D64161FF')
            plt.tick_params('y',colors='#D64161FF')
            plt.show()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")                                
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def NewWin(self):
        self.top = Toplevel(self.root)
        self.obj = PG2(self.top,self.trainfile,self.testfile)        
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Prep(self):
        if self.testfile != "" and self.trainfile != "" :
            command = self.NewWin()  
        else:
            messagebox.showerror("Error","Test Or Train File Is Missing")    
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#        






#------------------------------------------------PG2-------------------------------------------------#
class PG2():
    def __init__(self, root,trainfile,testfile):
        self.trainfile = trainfile
        self.testfile = testfile
        self.train=pd.read_csv(self.trainfile)
        self.root = root
        self.root.title("DATA PREPARATION")
        self.root.geometry("1350x700+0+0")
        self.im =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 1.jpg") 
        self.im1 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 3.png")
        self.im2 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 5.png")
        #self.im2 =ImageTk.PhotoImage(file="iiu.png")
        img1=Label(self.root,image=self.im).pack()
        frame =Frame(self.root,bg="#203a51")
        frame.place(x=375,y=0,height = 800,width = 580)
        txt1=Label(frame,text="Data Preperation",compound=LEFT,font=("times new roman",15),bg="#22394b",fg = "WHITE").grid(row=1,column=2,padx=0,pady=1)
        img2=Label(frame,image=self.im1,bd=0).grid(row=0,columnspan=5,pady=15)
        txt2=Button(frame,text="DataFrameHead",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.head).grid(row =2 ,column = 1,pady = 10)
        #txt3=Button(frame,text="info",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.info).grid(row =2 ,column = 2 ,pady = 10)
        txt5=Button(frame,text="DataSetDescription",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.describe).grid(row =2 ,column = 2 ,pady = 10)
        txt6=Button(frame,text="Nan In DataSet",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.isnull).grid(row =2 ,column = 3 ,pady = 10)
        txt7=Button(frame,text="RemovingNanValues",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.isnull2).grid(row =3 ,column = 1 ,pady = 10)
        txt8=Button(frame,text="DataTypesOriginal",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.dtypes).grid(row =3 ,column = 2,pady = 10)
        txt9=Button(frame,text="DataTypesChanged",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.dtypes2).grid(row =3 ,column = 3,pady = 10)
        txt10=Button(frame,text="Data Set",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.concate).grid(row =4 ,column = 1 ,pady = 10)
        txt11=Button(frame,text="Loan Status",width = 15,font=("times new roman",15),bg = "#22394b",fg = "WHITE",command = self.temp).grid(row =4 ,column = 3 ,pady = 10)
        img2=Label(frame,image=self.im2,bd=0).grid(row=5,columnspan=5,pady=15)
        txt17=Button(frame,text="Apply ML Models",width = 15,font=("times new roman",15),bg = "#1e6ead",fg = "WHITE",command =self.Prep).grid(row =6 ,column = 2 ,pady = 10) 
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def head(self):
        temp = self.train.head()
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\head.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\head.csv"')
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#
    
    def describe(self):
        temp = self.train.describe()
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\describe.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\describe.csv"')
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#    

    def isnull(self):
        temp = self.train.apply(lambda x:sum(x.isnull()),axis=0)
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\isnull.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\isnull.csv"')
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def isnull2(self):
        self.train['Gender'].fillna('Male',inplace=True)
        self.train['Married'].fillna('Yes',inplace=True)
        self.train['Dependents'].fillna(0,inplace=True)
        self.train['Self_Employed'].fillna('No',inplace=True)
        self.train['Credit_History'].fillna(1,inplace=True)
        self.train['LoanAmount'].fillna(self.train['LoanAmount'].mean(), inplace=True)
        self.train['Loan_Amount_Term'].fillna(360,inplace=True)
        temp = self.train.apply(lambda x:sum(x.isnull()),axis=0)
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\isnull2.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\isnull2.csv"')
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def dtypes(self):
        temp = self.train.dtypes
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\dtypes.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\dtypes.csv"')           
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def dtypes2(self):
        self.train['Dependents']=self.train['Dependents'].astype('str')
        var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
        le=LabelEncoder()
        for i in var_mod:
            self.train[i]=le.fit_transform(self.train[i].astype('str'))
        temp = self.train.dtypes
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\dtypes2.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\dtypes2.csv"')
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def concate(self):
        temp =pd.concat([self.train.iloc[:,1:6],self.train.iloc[:,8:12],self.train.iloc[:,13:14]],axis=1)
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\concate.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\concate.csv"')
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#    

    def temp(self):
        temp =self.train.iloc[:,12:13]
        temp.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\temp.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\temp.csv"') 
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def NewWin(self):
        self.top = Toplevel(self.root)
        self.obj = PG3(self.top,self.trainfile,self.testfile)        
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#

    def Prep(self):
            command = self.NewWin()  
    #--------------------------------------------------------------------------------------------------------#
    #--------------------------------------------------------------------------------------------------------#    






#------------------------------------------------------PG3----------------------------------------------------------#
class PG3():
    def __init__(self, root,trainfile,testfile):
        self.trainfile = trainfile
        self.testfile = testfile
        global temp5
        self.train=pd.read_csv(self.trainfile)
        self.root = root
        self.root.title("APPLY MACHINE LEARNING MODEL")
        self.root.geometry("1350x700+0+0")
        self.im =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 1.jpg") 
        self.im1 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 3.png")
        self.im2 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 5.png")
        self.temp15 = 4
        #self.im2 =ImageTk.PhotoImage(file="iiu.png")
        img1=Label(self.root,image=self.im).pack()
        frame =Frame(self.root,bg="#203a51")
        frame.place(x=375,y=0,height = 800,width = 580)
        img2=Label(frame,image=self.im1,bd=0).grid(row=0,columnspan=15,pady=15)
        #txt=Button(frame,text="MANUAL",width = 15,font=("times new roman",15),bg = "GREEN",fg = "WHITE",command =self.NewWin).grid(row =10 ,column = 2 ,pady = 10) 
        temp1 = self.train['Credit_History'].value_counts(ascending=True)
        temp2 = self.train.pivot_table(values='Loan_Status',index=['Credit_History'],aggfunc=lambda x: x.map({'Y':1,'N':0}).mean())
        self.train.apply(lambda x:sum(x.isnull()),axis=0)
        self.train['Gender'].fillna('Male',inplace=True)
        self.train['Married'].fillna('Yes',inplace=True)
        self.train['Dependents'].fillna(0,inplace=True)
        self.train['Self_Employed'].fillna('No',inplace=True)
        self.train['Credit_History'].fillna(1,inplace=True)
        self.train['LoanAmount'].fillna(self.train['LoanAmount'].mean(), inplace=True)
        self.train['Loan_Amount_Term'].fillna(360,inplace=True)
        self.train.apply(lambda x:sum(x.isnull()),axis=0)
        self.train['LoanAmount'] = np.log(self.train['LoanAmount'])
        self.train['TotalIncome'] = self.train['ApplicantIncome'] + self.train['CoapplicantIncome']
        self.train['TotalIncome'] = np.log(self.train['TotalIncome'])
        self.train['Dependents']=self.train['Dependents'].astype('str')
        var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area','Loan_Status']
        le=LabelEncoder()
        for i in var_mod:
            self.train[i]=le.fit_transform(self.train[i].astype('str'))
        X=pd.concat([self.train.iloc[:,1:6],self.train.iloc[:,8:12],self.train.iloc[:,13:14]],axis=1)
        Y=self.train.iloc[:,12:13]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.20, random_state=21)
        global Ynew
        Ynew = Y
        global Xnew
        Xnew = X
        #print(Xnew)
        #--------------------------------------------------------------------------------#
        #--------------------------------------------------------------------------------#

    
        accuracy = []
        precision = []
        recall = []
        #--------------------------------LogisticRegression------------------------------#
        modelLR=LogisticRegression(solver='liblinear')
        modelLR.fit(self.X_train,self.y_train.values.ravel())
        predictions1 = modelLR.predict(self.X_test)
        accuracy.append(metrics.accuracy_score(predictions1,self.y_test))
        precision.append(metrics.precision_score(predictions1,self.y_test))
        recall.append(metrics.recall_score(predictions1,self.y_test))
        #print(accuracy)
        #--------------------------------------------------------------------------------#
        #-------------------------------KNeighborsClassifier6----------------------------#
        modelKNN=KNeighborsClassifier(n_neighbors=6)
        predictor=['Credit_History','Gender','Married','Education']
        modelKNN.fit(self.X_train[predictor],self.y_train.values.ravel())
        predictions2 = modelKNN.predict(self.X_test[predictor])
        accuracy.append(metrics.accuracy_score(predictions2,self.y_test))
        precision.append(metrics.precision_score(predictions2,self.y_test))
        recall.append(metrics.recall_score(predictions2,self.y_test))
        #print(accuracy)
        #--------------------------------------------------------------------------------#
        #------------------------------KNeighborsClassifier10----------------------------#
        modelKNN=KNeighborsClassifier(n_neighbors=10)
        predictor=['Credit_History','Gender','Married','Education']
        modelKNN.fit(self.X_train[predictor],self.y_train.values.ravel())
        predictions3 = modelKNN.predict(self.X_test[predictor])
        accuracy.append(metrics.accuracy_score(predictions3,self.y_test))
        precision.append(metrics.precision_score(predictions3,self.y_test))
        recall.append(metrics.recall_score(predictions3,self.y_test))
        #print(accuracy)
        #--------------------------------------------------------------------------------#
        #------------------------------DecisionTreeClassifier----------------------------#
        modelDT=DecisionTreeClassifier()
        predictor=['Credit_History','Gender','Married','Education']
        modelDT.fit(self.X_train[predictor],self.y_train.values.ravel())
        predictions4 = modelDT.predict(self.X_test[predictor])
        accuracy.append(metrics.accuracy_score(predictions4,self.y_test))
        precision.append(metrics.precision_score(predictions4,self.y_test))
        recall.append(metrics.recall_score(predictions4,self.y_test))
        #print(accuracy)
        #--------------------------------------------------------------------------------#
        #---------------------------BaggingClassifier------------------------------------# 
        modelBKNN = BaggingClassifier(KNeighborsClassifier(),max_samples=0.50, max_features=0.5)
        predictor=['Education','Loan_Amount_Term','Credit_History','Dependents','Property_Area']
        modelBKNN.fit(self.X_train[predictor],self.y_train.values.ravel())
        predictions5 = modelBKNN.predict(self.X_test[predictor])
        accuracy.append(metrics.accuracy_score(predictions5,self.y_test))
        precision.append(metrics.precision_score(predictions5,self.y_test))
        recall.append(metrics.recall_score(predictions5,self.y_test))
        #print(accuracy)
        #--------------------------------------------------------------------------------#
        #---------------------------BaggingClassifier------------------------------------#
        modelBDT = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.50, max_features=0.5)
        predictor=['Education','Loan_Amount_Term','Credit_History','Dependents','Property_Area']
        modelBDT.fit(self.X_train[predictor],self.y_train.values.ravel())
        predictions6 = modelBDT.predict(self.X_test[predictor])
        accuracy.append(metrics.accuracy_score(predictions6,self.y_test))
        precision.append(metrics.precision_score(predictions6,self.y_test))
        recall.append(metrics.recall_score(predictions6,self.y_test))
        #print(accuracy)
        #-------------------------------------------------------------------------------#
        #----------------------------RandomForestClassifier-----------------------------#
        modelRF = RandomForestClassifier(n_estimators=50,min_samples_split=25,max_depth=7,max_features=1)
        modelRF.fit(self.X_train,self.y_train.values.ravel())
        predictions7 = modelRF.predict(self.X_test)
        accuracy.append(metrics.accuracy_score(predictions7,self.y_test))
        precision.append(metrics.precision_score(predictions7,self.y_test))
        recall.append(metrics.recall_score(predictions7,self.y_test))
        #print(accuracy)
        #-------------------------------------------------------------------------------#
        #---------------------------GaussianNB------------------------------------------#
        modelNB = GaussianNB()
        predictor=['Credit_History','Gender','Married','Education']
        modelNB.fit(self.X_train[predictor],self.y_train.values.ravel())
        predictions8 = modelNB.predict(self.X_test[predictor])
        accuracy.append(metrics.accuracy_score(predictions8,self.y_test))
        precision.append(metrics.precision_score(predictions8,self.y_test))
        recall.append(metrics.recall_score(predictions8,self.y_test))
        #print(accuracy)
        #-------------------------------------------------------------------------------#
        print(precision)
        print(recall)
        temp7 = sorted(accuracy, reverse=True)
        print(accuracy)
        i = 7
        temp8 = temp7[0]
        while i >= 0 :
            if temp8 == accuracy[i]:
                temp9 = i
                break
            i-=1    
        temp15 = i




        txt1=Label(frame,text="ML Models Accuracy :",compound=LEFT,font=("times new roman",20),bg="GREEN",fg = "WHITE").grid(row=1,column=0,padx=120,pady=5)
        txt1=Label(frame,text="L_REG :- " + str(accuracy[0]) ,compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=2,column=0,padx=100,pady=3)
        txt1=Label(frame,text="KNN_6 :- " + str(accuracy[1]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=3,column=0,padx=100,pady=3)
        txt1=Label(frame,text="KNN10 :- "+ str(accuracy[2]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=4,column=0,padx=100,pady=3)
        txt1=Label(frame,text="DTREE :- "+ str(accuracy[3]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=5,column=0,padx=100,pady=3)
        txt1=Label(frame,text="B_KNN :- "+ str(accuracy[4]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=6,column=0,padx=100,pady=3)
        txt1=Label(frame,text="BDTRE :- "+ str(accuracy[5]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=7,column=0,padx=100,pady=3)
        txt1=Label(frame,text="R_FOR :- "+ str(accuracy[6]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=8,column=0,padx=100,pady=3)
        txt1=Label(frame,text="N_BAY :- "+ str(accuracy[7]),compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=9,column=0,padx=100,pady=3)
        txt=Button(frame,text="APPLY",width = 15,font=("times new roman",15),bg = "GREEN",fg = "WHITE",command =self.ApplyMLModel).grid(row =10 ,column = 0 ,pady = 0) 
        self.acc = accuracy
        self.pre = precision
        self.rec = recall

    def ApplyMLModel(self):    
        self.top = Toplevel(self.root)
        self.obj = PG4(self.top,self.testfile,self.temp15,self.acc,self.pre,self.rec)












    #----------------------------------------PG4-----------------------------------------------------------------#

class PG4:         
    def __init__(self,root,testfile,temp15,accuracy,precision,recall):
        self.root = root
        self.testfile = testfile
        self.temp15 = temp15
        self.accuracy = accuracy
        self.recall = recall
        self.precision = precision
        #self.trainfile=trainfile
        #self.train=pd.read_csv(self.trainfile)
        test=pd.read_csv(self.testfile)
        self.root.title("ARTIFICIAL INTELLIGENCE")
        self.root.geometry("1350x700+0+0")
        self.im =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 1.jpg") 
        self.im1 =ImageTk.PhotoImage(file="C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Images\\PIC 3.png")
        img1=Label(self.root,image=self.im).pack()
        frame =Frame(self.root,bg="#203a51")
        frame.place(x=375,y=0,height = 800,width = 580)
        img2=Label(frame,image=self.im1,bd=0).grid(row=0,columnspan=15,pady=15)
        











    #---------------------------------------------------------------------------------------#
    #------------------------------  Data Cleaning Of Test File-----------------------------#
        test['Gender'].fillna('Male',inplace=True)
        test['Dependents'].fillna(0,inplace=True)
        test['Self_Employed'].fillna('No',inplace=True)
        test['Credit_History'].fillna(1,inplace=True)
        test['LoanAmount'].fillna(test['LoanAmount'].mean(),inplace=True)
        test['Loan_Amount_Term'].fillna(360,inplace=True)
        test.apply(lambda x:sum(x.isnull()),axis=0)
        test['TotalIncome'] = test['ApplicantIncome'] + test['CoapplicantIncome']
        X1_test=pd.concat([test.iloc[:,1:6],test.iloc[:,8:]],axis=1)
        X1_test['Dependents']=X1_test['Dependents'].astype('str')
        var_mod=['Gender','Married','Dependents','Education','Self_Employed','Property_Area']
        le=LabelEncoder()
        for i in var_mod:
            X1_test[i]=le.fit_transform(X1_test[i].astype('str'))
    #---------------------------------------------------------------------------------------#
    #---------------------------------------------------------------------------------------#
        temp7 = sorted(self.accuracy, reverse=True)
        i = 7
        temp8 = temp7[0]
        while i >= 0 :
            if temp8 == self.accuracy[i]:
                temp15 = i
                break
            i-=1    
        if temp15 == 0:
            temp10 = "Logistic Regression"
        elif temp15 == 1:
            temp10 = "KNN with 6 neighbors"
        elif temp15 == 2:
            temp10 = "KNN with 10 neighbors"        
        elif temp15 == 3:
            temp10 = "Decision Tree"
        elif temp15 == 4:
            temp10 = "Bagging using KNN"
        elif temp15 == 5:
            temp10 = "Bagging using Decision Tree"
        elif temp15 == 6:
            temp10 = "Random Forest"
        elif temp15 == 7:
            temp10 = "Naïve Baize"                    
    #-----------------------------------------------------------------------------------#
    #-----------------------------------------------------------------------------------#

    #--------------------------------LogisticRegression------------------------------#
        if temp15 == 0:
            modelLR=LogisticRegression(solver='liblinear')
            modelLR.fit(Xnew,Ynew.values.ravel())
            predictions = modelLR.predict(X1_test)
            predictions = predictions.astype('str')
        #--------------------------------------------------------------------------------#
        #-------------------------------KNeighborsClassifier6----------------------------#
        elif temp15 == 1:
            modelKNN=KNeighborsClassifier(n_neighbors=6)
            predictor=['Credit_History','Gender','Married','Education']
            modelKNN.fit(Xnew[predictor],Ynew.values.ravel())
            predictions = modelKNN.predict(X1_test[predictor])
            predictions = predictions.astype('str')
        #--------------------------------------------------------------------------------#
        #------------------------------KNeighborsClassifier10----------------------------#
        elif temp15 == 2:    
            modelKNN=KNeighborsClassifier(n_neighbors=10)
            predictor=['Credit_History','Gender','Married','Education']
            modelKNN.fit(Xnew[predictor],Ynew.values.ravel())
            predictions = modelKNN.predict(X1_test[predictor])
            predictions = predictions.astype('str')
        #--------------------------------------------------------------------------------#
        #------------------------------DecisionTreeClassifier----------------------------#
        elif temp15 == 3:    
            modelDT=DecisionTreeClassifier()
            predictor=['Credit_History','Gender','Married','Education']
            modelDT.fit(Xnew[predictor],Ynew.values.ravel())
            predictions = modelDT.predict(X1_test[predictor])
            predictions = predictions.astype('str')
        #--------------------------------------------------------------------------------#
        #---------------------------BaggingClassifier------------------------------------# 
        elif temp15 == 4:    
            modelBKNN = BaggingClassifier(KNeighborsClassifier(),max_samples=0.50, max_features=0.5)
            predictor=['Education','Loan_Amount_Term','Credit_History','Dependents','Property_Area']
            modelBKNN.fit(Xnew[predictor],Ynew.values.ravel())
            predictions = modelBKNN.predict(X1_test[predictor])
            predictions = predictions.astype('str')
        #--------------------------------------------------------------------------------#
        #---------------------------BaggingClassifier------------------------------------#
        elif temp15 == 5:
            modelBDT = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.50, max_features=0.5)
            predictor=['Education','Loan_Amount_Term','Credit_History','Dependents','Property_Area']
            modelBDT.fit(Xnew[predictor],Ynew.values.ravel())
            predictions = modelBDT.predict(X1_test[predictor])
            predictions = predictions.astype('str')
        #-------------------------------------------------------------------------------#
        #----------------------------RandomForestClassifier-----------------------------#
        elif temp15 == 6:
            modelRF = RandomForestClassifier(n_estimators=50,min_samples_split=25,max_depth=7,max_features=1)
            modelRF.fit(Xnew,Ynew.values.ravel())
            predictions = modelRF.predict(X1_test)
            predictions = predictions.astype('str')
        #-------------------------------------------------------------------------------#
        #---------------------------GaussianNB------------------------------------------#
        elif temp15 == 7:
            modelNB = GaussianNB()
            predictor=['Credit_History','Gender','Married','Education']
            modelNB.fit(Xnew[predictor],Ynew.values.ravel())
            predictions = modelNB.predict(X1_test[predictor])
            predictions = predictions.astype('str')

        #------------------------------------------------------------------------------#
        #------------------------------------------------------------------------------#
        for i in range(0,367):
            if(predictions[i]=='1') :
                predictions[i]='Y'
            elif(predictions[i]=='0'):
                predictions[i]='N'    
        
        self.df=pd.concat([test['Loan_ID'],pd.DataFrame({'Loan_Status':predictions})],axis=1)
        
        txt1=Label(frame,text="Model Selected:- "+ temp10,compound=LEFT,font=("times new roman",20),bg="#203a51",fg = "WHITE").grid(row=1,column=0,padx=100,pady=3)
        txt=Button(frame,text="Accuracy Graph",width = 15,font=("times new roman",15),bg = "GREEN",fg = "WHITE",command =self.ApplyMLModel).grid(row =2,column = 0 ,padx=200,pady = 20) 
        txt=Button(frame,text="Precision Graph",width = 15,font=("times new roman",15),bg = "GREEN",fg = "WHITE",command =self.ApplyMLModel1).grid(row =3,column = 0 ,padx=200,pady = 20) 
        txt=Button(frame,text="Recall Graph",width = 15,font=("times new roman",15),bg = "GREEN",fg = "WHITE",command =self.ApplyMLModel2).grid(row =4,column = 0 ,padx=200,pady = 20) 
        txt=Button(frame,text="Save&OpenPrediction",width = 15,font=("times new roman",15),bg = "GREEN",fg = "WHITE",command =self.savefile).grid(row =5,column = 0 ,padx=200,pady = 20) 
        print(predictions)

    def ApplyMLModel(self):
        
        t = ['LR','KNN6','KNN10','DTC','BC1','BC2','RFC','GNB']
        t=np.array(t)
        self.accuracy=np.array(self.accuracy)

        fig , ax = plt.subplots()
        ax.plot(t,self.accuracy,color = 'GREEN',marker = 'o',linestyle = '--')
        ax.set_xlabel('Machine Learning Models', color='GREEN')
        ax.set_ylabel('accuracy Of Each Model', color='r')
        ax.tick_params('x',colors='GREEN')
        ax.tick_params('y',colors='r')
        plt.show()
    def ApplyMLModel1(self):
        
        t = ['LR','KNN6','KNN10','DTC','BC1','BC2','RFC','GNB']
        t=np.array(t)
        self.precision=np.array(self.precision)

        fig , ax = plt.subplots()
        ax.plot(t,self.precision,color = 'GREEN',marker = 'o',linestyle = '--')
        ax.set_xlabel('Machine Learning Models', color='GREEN')
        ax.set_ylabel('Precision Of Each Model', color='r')
        ax.tick_params('x',colors='GREEN')
        ax.tick_params('y',colors='r')
        plt.show()
            
    def ApplyMLModel2(self): 
        t = ['LR','KNN6','KNN10','DTC','BC1','BC2','RFC','GNB']
        t=np.array(t)
        self.recall=np.array(self.recall)
        fig , ax = plt.subplots()
        ax.plot(t,self.recall,color = 'GREEN',marker = 'o',linestyle = '--')
        ax.set_xlabel('Machine Learning Models', color='GREEN')
        ax.set_ylabel('Recall Of Each Model', color='r')
        ax.tick_params('x',colors='GREEN')
        ax.tick_params('y',colors='r')
        plt.show()    

    def savefile(self):
        self.df.to_csv("C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\PredictionResult.csv")
        os.system('start "excel" "C:\\Users\\z1736\\Desktop\\Artificial Intelligence Project\\Output Files\\PredictionResult.csv"') 

        #txt1=Label(frame,text=temp10 ,compound=LEFT,font=("times new roman",20),bg="#1e6ead",fg = "WHITE").grid(row=1,column=1,padx=0,pady=1)
        #txt1=Label(frame,text=temp15 ,compound=LEFT,font=("times new roman",20),bg="#1e6ead",fg = "WHITE").grid(row=2,column=1,padx=0,pady=1)
        #txt1=Button(frame,text="Hye",compound=LEFT,font=("times new roman",20),bg="#1e6ead",fg = "WHITE").grid(row=3,column=1,padx=0,pady=1)






if __name__ == '__main__':
    main()





