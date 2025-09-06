# Exno:1
Data Cleaning Process

# AIM
To read the given data and perform data cleaning and save the cleaned data to a file.

# Explanation
Data cleaning is the process of preparing data for analysis by removing or modifying data that is incorrect ,incompleted , irrelevant , duplicated or improperly formatted. Data cleaning is not simply about erasing data ,but rather finding a way to maximize datasets accuracy without necessarily deleting the information.

# Algorithm
STEP 1: Read the given Data

STEP 2: Get the information about the data

STEP 3: Remove the null values from the data

STEP 4: Save the Clean data to the file

STEP 5: Remove outliers using IQR

STEP 6: Use zscore of to remove outliers

# Coding and Output
```
import pandas as pd
df=pd.read_csv("SAMPLEIDS.csv")
df
```

<img width="941" height="833" alt="data science exp1 0 1" src="https://github.com/user-attachments/assets/064cc01c-ae69-468b-9567-c5f7631c29c7" />

```
df.head()
```

<img width="907" height="291" alt="data science exp1 0 2" src="https://github.com/user-attachments/assets/3eced306-2376-495e-8bd7-c37181a0f542" />

```
df.tail()
```

<img width="925" height="260" alt="data science exp1 0 3" src="https://github.com/user-attachments/assets/e8d515f3-b43d-4c98-a8fc-a1b93522f9d7" />

```
df.isnull()
```

<img width="758" height="748" alt="data science exp1 0 4" src="https://github.com/user-attachments/assets/69baa510-3467-4136-9c7b-bcdcc96dc6e6" />

```
df.notnull()
```

<img width="757" height="736" alt="data science exp1 0 5" src="https://github.com/user-attachments/assets/fa90f088-dc90-4d63-8aae-cc7b14b66c77" />

```
df.isnull().sum()
```

<img width="281" height="492" alt="data science exp1 0 6" src="https://github.com/user-attachments/assets/5353a88f-203a-4adc-9239-e385f7220d67" />

```
df.isnull().any()
```

<img width="247" height="487" alt="data science exp1 0 7" src="https://github.com/user-attachments/assets/56ebf357-6521-4977-90ab-472fd9ee79d3" />

```
df.dropna(axis=0)
```

<img width="917" height="478" alt="data science exp1 0 8" src="https://github.com/user-attachments/assets/fa662eb1-5b75-4660-8c1e-91525cda2eb2" />

```
df.dropna(axis=1)
```
<img width="297" height="742" alt="data science exp1 0 9" src="https://github.com/user-attachments/assets/1d3166cf-4a7c-42dc-a0d4-295f71a16721" />

```
df.fillna(7)
```

<img width="907" height="752" alt="data science exp1 0 10" src="https://github.com/user-attachments/assets/5344bf34-d34e-4ae3-9651-88b60405cc13" />

```
df.fillna(method='ffill')
```

<img width="1381" height="768" alt="data science exp1 0 11" src="https://github.com/user-attachments/assets/b565d540-77d0-4f61-a1da-fd7c384c43e9" />

```
df.fillna(method='bfill')
```

<img width="1388" height="772" alt="data science exp1 0 12" src="https://github.com/user-attachments/assets/16ead2a5-33bb-4b27-a13d-ba4d4f3efb81" />

```
df.fillna({'GENDER':'MALE','NAME':'SANTHOSH','ADDRESS':'KANCHIPURAM','M1':'76.0','M2':'69.0','M3':'80.0','M4':'76.0','TOTAL':'301.0','AVG':'100.333333'})
```

<img width="1222" height="741" alt="data science exp1 0 13" src="https://github.com/user-attachments/assets/27e8561a-66ec-4984-8a0f-2482b38d7b25" />

```
import pandas as pd
ir=pd.read_csv("/content/iris.csv")
ir
```

<img width="637" height="500" alt="data science exp1 0 14" src="https://github.com/user-attachments/assets/c60bf661-0290-42f5-8e28-999bd8c1aff2" />

```
ir.describe()
```

<img width="562" height="330" alt="data science exp1 0 15" src="https://github.com/user-attachments/assets/c7f60394-9674-4514-974a-d7969a54ed94" />

```
import seaborn as sns
sns.boxplot(x='sepal_width',data=ir)
```

<img width="588" height="522" alt="data science exp1 0 16" src="https://github.com/user-attachments/assets/5e740af0-8005-44d4-8ba2-c0d33f44f3f1" />

```
Q1=ir.sepal_width.quantile(0.25)
Q3=ir.sepal_width.quantile(0.75)
IQR=Q3-Q1
print(IQR)
```

<img width="333" height="127" alt="data science exp1 0 17" src="https://github.com/user-attachments/assets/71106ffd-b44a-479c-b07f-e78af4ba9ab6" />

```
rid=ir[((ir.sepal_width<(Q1-1.5*IQR))|(ir.sepal_width>(Q3+1.5*IQR)))]
rid['sepal_width']
```

<img width="586" height="253" alt="data science exp1 0 18" src="https://github.com/user-attachments/assets/b174e9a1-8a65-4954-807c-b22f9c37f8c9" />

```
delid=ir[~((ir.sepal_width<(Q1-1.5*IQR))|(ir.sepal_width>(Q3+1.5*IQR)))]
delid
```

<img width="652" height="482" alt="data science exp1 0 19" src="https://github.com/user-attachments/assets/8afcc5e8-e97d-41d4-b920-5fabeec4ea95" />

```
sns.boxplot(x='sepal_width',data=delid
```

<img width="606" height="516" alt="data science exp1 0 20" src="https://github.com/user-attachments/assets/b8d234aa-ddff-4d08-bd60-41d825dd86ea" />

```
import numpy as np
import scipy.stats as stats
xyz=np.abs(stats.zscore(ir['sepal_width']))
xyz
```

<img width="621" height="637" alt="data science exp1 0 21" src="https://github.com/user-attachments/assets/79212126-c02e-4229-a272-b38fff9351e5" />

```
ir1=ir[xyz>3]
ir1
```

<img width="625" height="157" alt="data science exp1 0 22" src="https://github.com/user-attachments/assets/feb03059-f5b8-4c34-a904-41ad5ca14f6b" />



























# Result
          The given data and perform data cleaning and save the cleaned data to a file is completed successfully.
