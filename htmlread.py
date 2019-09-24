from __future__ import division, unicode_literals
import codecs
from bs4 import BeautifulSoup

def Ali(noodle):
    file = open("Ali.txt", "w")
    name=""
    inter=""
    school=""
    go = False
    teach = False
    email="ma04@txstate.edu"
    for i in noodle.split():
        if name != "Moonis Ali " and i == "Moonis" or i == "Ali":
            name=name+i+" "

        if teach == True:
            school=school+i+" "
        if i == "Office":
            teach = False
            
        
        if go == True:
            inter=inter+i+" "
        if i == "Interests":
            go = True
        if i == "Education":
            go = False
            teach = True
    sliced=name.split()
    first=sliced[0]
    last=sliced[1]
    name=first + " " + last
    file.write("Name: ")
    file.write(name)
    file.write("\n")
    file.write(school)
    file.write("\nResearch Interests: ")    
    file.write(inter)
    file.write("\nEmail: ")
    file.write(email)    
    file.close()
    
def Gao(broth):
    file = open("Gao.txt", "w")
    
    name=""
    inter=""
    school=""
    go = False
    teach = False
    email="jg66@txstate.edu"
    for i in broth.split():
        if i == "Byron" or i == "Gao":
            name=name+i+" "

        if teach == True:
            school=school+i+" "
        if i == "Office":
            teach = False
            
        if go == True:
            inter=inter+i+" "
        if i == "Interests":
            go = True
        if i == "Education":
            go = False
            teach = True
    sliced=name.split()
    first=sliced[0]
    last=sliced[1]
    name=first + " " + last
    file.write("Name: ")
    file.write(name)
    file.write("\nEducation: ")
    file.write(school)
    file.write("\nResearch Interests: ")    
    file.write(inter)
    file.write("\nEmail: ")
    file.write(email)    
    file.close()
    
def Koh(chili):
    file = open("Koh.txt", "w")
    name=""
    inter=""
    school=""
    go = False
    teach = False
    email="lk04@txstate.edu"
    for i in chili.split():
        if name != "Lee-Song Koh " and i == "Lee-Song" or i == "Koh":
            name=name+i+" "

        if teach == True:
            school=school+i+" "
        if i == "Office":
            teach = False
            
        if go == True:
            inter=inter+i+" "
        if i == "Interests":
            go = True
        if i == "Education":
            go = False
            teach = True
    sliced=name.split()
    first=sliced[0]
    last=sliced[1]
    name=first + " " + last
    file.write("Name: ")
    file.write(name)
    file.write("\nEducation: ")
    file.write(school)
    file.write("\nResearch Interests: ")    
    file.write(inter)
    file.write("\nEmail: ")
    file.write(email)    
    file.close()
    
"Main"
f=codecs.open("Ali.html", 'r', 'utf-8')
inF=codecs.open("Gao.html", 'r', 'utf-8')
file=codecs.open("Koh.html", 'r', 'utf-8')
soup=BeautifulSoup(f.read()).get_text()
beat=BeautifulSoup(inF.read()).get_text()
vegi=BeautifulSoup(file.read()).get_text()
noodle=str(soup)
broth=str(beat)
chili=str(vegi)
Ali(noodle)
Gao(broth)
Koh(chili)
