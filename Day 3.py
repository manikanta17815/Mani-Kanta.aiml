#!/usr/bin/env python
# coding: utf-8

# In[1]:


sub1 = int(input("Enter marks of subject-01 : " ))
sub2 = int(input("Enter marks of subject-02 : " ))
sub3 = int(input("Enter marks of subject-03 : " ))
avg = (sub1 + sub2 + sub3)/3
if avg >= 90:
    print("Grade: A")
elif 80 <= avg < 90:  
    print("Grade: B")
elif 70 <= avg < 80:  
    print("Grade: C")
else :
    print("Grade : Fail")


# In[ ]:




