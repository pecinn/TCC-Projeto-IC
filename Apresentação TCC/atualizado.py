# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:55:57 2019

@author: João Felipe
"""

import sys
from PyQt5 import uic, QtGui
from PyQt5.QtWidgets import QFileDialog, QApplication, QWidget, QLabel, QTableWidget, QTableWidgetItem, QListWidget, QListWidgetItem
import pandas as pd
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
from scipy import stats


Ui_MainWindow, QtBaseClass = uic.loadUiType("interface.ui") 
 

class MainWindow(QWidget, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)
        self.botaoPesquisar.clicked.connect(self.abrirBase)
        self.botaoClassificar.clicked.connect(self.classificarBase)
     
        
    def abrirBase(self):
       fileName= QFileDialog.getOpenFileName(self, 'Open file',"","CSV Files (*.csv)")
       self.dbInput.setText(fileName[0])
    
   
    def classificarBase(self):
        
        # carrega dataset
        dados = pd.read_csv(self.dbInput.text(), sep=';')
        
        #Pega a coluna de classes e joga na variavel y
        df = list(dados.columns.values)
        colunaClasse = df[-1]
        y= dados.pop(colunaClasse).values
        
        # x = carrega as instâncias
        scaler = preprocessing.StandardScaler().fit(dados.values)
        x=scaler.transform(dados.values)
        
        cont_knn = 0
        cont_arvores = 0
        cont_nb = 0
        list_knn=[]
        list_arvores=[]
        list_nb=[]
        
        if self.radioCross.isChecked():
            
            # define folds
            folds = int(self.inputFold.text())
            kf = KFold(n_splits=folds, shuffle=True)
            
            for treino_index, teste_index in kf.split(x):
                x_treino, x_teste = x[treino_index], x[teste_index]
                y_treino, y_teste = y[treino_index], y[teste_index]
                
                self.listAcuracia.clear()
                self.listAcuracia.addItem("----------Cross Validation--------")
                self.listAcuracia.addItem("Acurácia Geral (Taxa de Acerto)")
                
                #knn escolhido
                if self.cbKnn.isChecked() and self.inputK.text()!="" :
                    cont_knn = 1
                    #vizinhos, o usuario escolhe o valor de k
                    k = int(self.inputK.text())
                    classificador_KNN = KNeighborsClassifier(n_neighbors=k)
                    #list_knn=[]
                    #KNN
                    #cria o modelo do KNN = treinamento
                    classificador_KNN.fit(x_treino, y_treino)
                    # classifica os dados do x_teste
                    y_predict=classificador_KNN.predict(x_teste)
                    #adiciona a acurácia na lista do Knn
                    list_knn.append(accuracy_score(y_teste, y_predict, 
                                                  normalize=True))
                    self.listAcuracia.addItem("KNN - " +str(round(np.mean(list_knn)*100, 2))+" %")
                    
                  
                if self.cbArvores.isChecked() :  
                    cont_arvores = 1
                    classificador_arvores = DecisionTreeClassifier(random_state=0)
                    #Arvore
                    #cria o modelo do Arvore = treinamento
                    classificador_arvores.fit(x_treino, y_treino)
                    # classifica os dados do X_teste
                    y_predict=classificador_arvores.predict(x_teste)
                    #adiciona a acurácia na lista do arvore
                    list_arvores.append(accuracy_score(y_teste, y_predict, 
                                                      normalize=True))
                    self.listAcuracia.addItem("C4.5 - " +str(round(np.mean(list_arvores)*100, 2))+" %")
                       
                if self.cbNb.isChecked() : 
                    cont_nb = 1
                    classificador_Nb = GaussianNB()
                    #NB
                    #cria o modelo do NB = treinamento
                    classificador_Nb.fit(x_treino, y_treino)
                    #ckassufuca is dadis do x_teste
                    y_predict=classificador_Nb.predict(x_teste)
                    #adiciona a acurácia na lista da arvore
                    list_nb.append(accuracy_score(y_teste, y_predict, 
                                                     normalize=True))
                    self.listAcuracia.addItem("Naive Bayes - " +str(round(np.mean(list_nb)*100, 2))+" %")
                self.tablePValue.clear()        
                self.tablePValue.setRowCount(3)
                self.tablePValue.setColumnCount(3)
                self.tablePValue.setItem(0,0, QTableWidgetItem("P-Values"))
                
                if cont_knn==1 and cont_arvores==1:
                    knn_arvores = stats.ttest_rel(list_knn,list_arvores)
                    pvalue1 = knn_arvores[1]
                    self.tablePValue.setItem(1,0, QTableWidgetItem("KNN x C4.5"))
                    self.tablePValue.setItem(2,0, QTableWidgetItem(str(round(pvalue1, 3))))
                    
                    
                    
                if cont_knn==1 and cont_nb==1:
                    knn_nb = stats.ttest_rel(list_knn,list_nb)
                    pvalue2 = knn_nb[1]
                    self.tablePValue.setItem(1,1, QTableWidgetItem("KNN x NB"))
                    self.tablePValue.setItem(2,1, QTableWidgetItem(str(round(pvalue2, 3))))
                    
            
                        
                if cont_arvores==1 and cont_nb==1:
                    arvores_nb = stats.ttest_rel(list_arvores,list_nb)
                    pvalue3 = arvores_nb[1]
                    self.tablePValue.setItem(1,2, QTableWidgetItem("C4.5 x NB"))
                    self.tablePValue.setItem(2,2, QTableWidgetItem(str(round(pvalue3, 3))))
                    
        else:
                 
                 teste = int(self.inputHoldout.text()) / 100
                 
                 x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=teste)
                 
                 self.tablePValue.clear()      
                 self.listAcuracia.clear()
                 self.listAcuracia.addItem("----------Holdout--------")
                 self.listAcuracia.addItem("Acurácia Geral (Taxa de Acerto)")
                 
                 #knn escolhido
                 if self.cbKnn.isChecked() and self.inputK.text()!="" :
                         cont_knn = 1
                         #vizinhos, o usuario escolhe o valor de k
                         k = int(self.inputK.text())
                         classificador_KNN = KNeighborsClassifier(n_neighbors=k)
                         classificador_KNN.fit(x_train, y_train)
                         y_predict = classificador_KNN.predict(x_test)
                         list_knn.append(accuracy_score(y_test, y_predict, normalize=True))
                         self.listAcuracia.addItem("KNN - " +str(round(np.mean(list_knn)*100, 2))+" %")
                     
                 #c4.5 escolhido
                 if self.cbArvores.isChecked() :  
                         cont_arvores = 1
                         classificador_arvores = DecisionTreeClassifier(random_state=0)
                         #Arvore
                         #cria o modelo do Arvore = treinamento
                         classificador_arvores.fit(x_train, y_train)
                         # classifica os dados do X_teste
                         y_predict=classificador_arvores.predict(x_test)
                         #adiciona a acurácia na lista do arvore
                         list_arvores.append(accuracy_score(y_test, y_predict, 
                                                           normalize=True))
                         self.listAcuracia.addItem("C4.5 - " +str(round(np.mean(list_arvores)*100, 2))+" %")
                         
                 if self.cbNb.isChecked() : 
                         cont_nb = 1
                         classificador_Nb = GaussianNB()
                         #NB
                         #cria o modelo do NB = treinamento
                         classificador_Nb.fit(x_train, y_train)
                         #ckassufuca is dadis do x_teste
                         y_predict=classificador_Nb.predict(x_test)
                         #adiciona a acurácia na lista da arvore
                         list_nb.append(accuracy_score(y_test, y_predict, 
                                                          normalize=True))
                         self.listAcuracia.addItem("Naive Bayes - " +str(round(np.mean(list_nb)*100, 2))+" %")
                        
                       
                 
def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
   main()