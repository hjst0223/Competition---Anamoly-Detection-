# EDA

## Import Libraries


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as img
import seaborn as sb
```

## Raw Data Loading


```python
train_df = pd.read_csv('./data/train_df.csv')
display(train_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>file_name</th>
      <th>class</th>
      <th>state</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>10000.png</td>
      <td>transistor</td>
      <td>good</td>
      <td>transistor-good</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>10001.png</td>
      <td>capsule</td>
      <td>good</td>
      <td>capsule-good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>10002.png</td>
      <td>transistor</td>
      <td>good</td>
      <td>transistor-good</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>10003.png</td>
      <td>wood</td>
      <td>good</td>
      <td>wood-good</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>10004.png</td>
      <td>bottle</td>
      <td>good</td>
      <td>bottle-good</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>4272</th>
      <td>4272</td>
      <td>14272.png</td>
      <td>transistor</td>
      <td>good</td>
      <td>transistor-good</td>
    </tr>
    <tr>
      <th>4273</th>
      <td>4273</td>
      <td>14273.png</td>
      <td>transistor</td>
      <td>good</td>
      <td>transistor-good</td>
    </tr>
    <tr>
      <th>4274</th>
      <td>4274</td>
      <td>14274.png</td>
      <td>grid</td>
      <td>good</td>
      <td>grid-good</td>
    </tr>
    <tr>
      <th>4275</th>
      <td>4275</td>
      <td>14275.png</td>
      <td>zipper</td>
      <td>good</td>
      <td>zipper-good</td>
    </tr>
    <tr>
      <th>4276</th>
      <td>4276</td>
      <td>14276.png</td>
      <td>screw</td>
      <td>good</td>
      <td>screw-good</td>
    </tr>
  </tbody>
</table>
<p>4277 rows × 5 columns</p>
</div>



- 총 4277개의 train 데이터

## Pillow를 이용한 이미지 처리


```python
from PIL import Image

img = Image.open('./data/train/10000.png')  # png이므로 channel이 4

## 해당 이미지 파일에 대한 이미지 객체를 들고 옴
print(type(img))

plt.imshow(img)
plt.show()
```

    <class 'PIL.PngImagePlugin.PngImageFile'>
    


    
![png](output_6_1.png)
    



```python
pixel = np.array(img)  # pillow 이미지 객체를 이용해서 ndarray 생성
print(pixel)
```

    [[[141 100  95]
      [143 100  92]
      [145 102  90]
      ...
      [138 103  92]
      [135  98  92]
      [135  99  92]]
    
     [[147 102  93]
      [145 102  90]
      [145 105  88]
      ...
      [147 104  91]
      [141 102  94]
      [137 101  93]]
    
     [[143 102  92]
      [146 104  89]
      [149 103  86]
      ...
      [146 105  89]
      [142 104  93]
      [139 102  94]]
    
     ...
    
     [[145  98  89]
      [142  99  88]
      [141  99  87]
      ...
      [151 107  92]
      [150 109  93]
      [150 107  96]]
    
     [[146 104  96]
      [147 103  90]
      [147 103  85]
      ...
      [158 109  94]
      [158 110  93]
      [157 109  97]]
    
     [[136  95  91]
      [140  96  85]
      [141  99  82]
      ...
      [152 102  89]
      [149 102  87]
      [149 103  91]]]
    
    
```python
train_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 4277 entries, 0 to 4276
    Data columns (total 5 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   index      4277 non-null   int64 
     1   file_name  4277 non-null   object
     2   class      4277 non-null   object
     3   state      4277 non-null   object
     4   label      4277 non-null   object
    dtypes: int64(1), object(4)
    memory usage: 167.2+ KB
    


```python
classList = train_df['class'].unique()
classList
```




    array(['transistor', 'capsule', 'wood', 'bottle', 'screw', 'cable',
           'carpet', 'hazelnut', 'pill', 'metal_nut', 'zipper', 'leather',
           'toothbrush', 'tile', 'grid'], dtype=object)




```python
train_df['class'].nunique()
```




    15



- 총 15개의 class


```python
train_df['state'].unique()
```




    array(['good', 'bent_wire', 'hole', 'pill_type', 'scratch', 'thread_side',
           'fabric_border', 'crack', 'manipulated_front', 'contamination',
           'split_teeth', 'combined', 'color', 'thread_top', 'missing_cable',
           'squeeze', 'rough', 'poke', 'flip', 'metal_contamination',
           'bent_lead', 'fabric_interior', 'fold', 'glue_strip',
           'scratch_neck', 'scratch_head', 'cut', 'broken_large',
           'broken_small', 'cut_outer_insulation', 'squeezed_teeth',
           'defective', 'cut_inner_insulation', 'missing_wire', 'thread',
           'broken', 'faulty_imprint', 'glue', 'damaged_case', 'gray_stroke',
           'bent', 'print', 'broken_teeth', 'oil', 'misplaced', 'cable_swap',
           'poke_insulation', 'cut_lead', 'liquid'], dtype=object)




```python
train_df['state'].nunique()
```




    49



- 총 49개의 state


```python
train_df['label'].unique()
```




    array(['transistor-good', 'capsule-good', 'wood-good', 'bottle-good',
           'screw-good', 'cable-bent_wire', 'carpet-hole', 'hazelnut-good',
           'pill-pill_type', 'cable-good', 'metal_nut-scratch', 'pill-good',
           'screw-thread_side', 'zipper-fabric_border', 'leather-good',
           'pill-scratch', 'toothbrush-good', 'hazelnut-crack',
           'screw-manipulated_front', 'zipper-good', 'tile-good',
           'carpet-good', 'metal_nut-good', 'bottle-contamination',
           'grid-good', 'zipper-split_teeth', 'pill-crack', 'wood-combined',
           'pill-color', 'screw-thread_top', 'cable-missing_cable',
           'capsule-squeeze', 'zipper-rough', 'capsule-crack', 'capsule-poke',
           'metal_nut-flip', 'carpet-metal_contamination', 'metal_nut-color',
           'transistor-bent_lead', 'zipper-fabric_interior', 'leather-fold',
           'tile-glue_strip', 'screw-scratch_neck', 'screw-scratch_head',
           'hazelnut-cut', 'bottle-broken_large', 'bottle-broken_small',
           'leather-cut', 'cable-cut_outer_insulation',
           'zipper-squeezed_teeth', 'toothbrush-defective',
           'cable-cut_inner_insulation', 'pill-contamination',
           'cable-missing_wire', 'carpet-thread', 'grid-broken',
           'pill-faulty_imprint', 'hazelnut-hole', 'leather-glue',
           'leather-poke', 'transistor-damaged_case', 'wood-scratch',
           'tile-gray_stroke', 'capsule-faulty_imprint', 'grid-glue',
           'zipper-combined', 'carpet-color', 'grid-bent', 'pill-combined',
           'hazelnut-print', 'cable-combined', 'capsule-scratch',
           'metal_nut-bent', 'zipper-broken_teeth', 'tile-oil',
           'transistor-misplaced', 'grid-thread', 'grid-metal_contamination',
           'carpet-cut', 'wood-color', 'cable-cable_swap', 'tile-crack',
           'leather-color', 'cable-poke_insulation', 'transistor-cut_lead',
           'wood-hole', 'tile-rough', 'wood-liquid'], dtype=object)




```python
train_df['label'].nunique()
```




    88



- 총 88개의 label


```python
test_df = pd.read_csv('./data/test_df.csv')
display(test_df)
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>index</th>
      <th>file_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>20000.png</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>20001.png</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>20002.png</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>20003.png</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>20004.png</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2149</th>
      <td>2149</td>
      <td>22149.png</td>
    </tr>
    <tr>
      <th>2150</th>
      <td>2150</td>
      <td>22150.png</td>
    </tr>
    <tr>
      <th>2151</th>
      <td>2151</td>
      <td>22151.png</td>
    </tr>
    <tr>
      <th>2152</th>
      <td>2152</td>
      <td>22152.png</td>
    </tr>
    <tr>
      <th>2153</th>
      <td>2153</td>
      <td>22153.png</td>
    </tr>
  </tbody>
</table>
<p>2154 rows × 2 columns</p>
</div>



```python
test_df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2154 entries, 0 to 2153
    Data columns (total 2 columns):
     #   Column     Non-Null Count  Dtype 
    ---  ------     --------------  ----- 
     0   index      2154 non-null   int64 
     1   file_name  2154 non-null   object
    dtypes: int64(1), object(1)
    memory usage: 33.8+ KB



## Preprocessing


```python
train_df.isnull().sum()
```




    index        0
    file_name    0
    class        0
    state        0
    label        0
    dtype: int64



- 결측치 없음

### 1. label별 개수


```python
train_df[['class', 'label']].groupby('label').count().rename(columns={'class':'count'})
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bottle-broken_large</th>
      <td>10</td>
    </tr>
    <tr>
      <th>bottle-broken_small</th>
      <td>11</td>
    </tr>
    <tr>
      <th>bottle-contamination</th>
      <td>11</td>
    </tr>
    <tr>
      <th>bottle-good</th>
      <td>209</td>
    </tr>
    <tr>
      <th>cable-bent_wire</th>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>zipper-fabric_interior</th>
      <td>8</td>
    </tr>
    <tr>
      <th>zipper-good</th>
      <td>240</td>
    </tr>
    <tr>
      <th>zipper-rough</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-split_teeth</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-squeezed_teeth</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>88 rows × 1 columns</p>
</div>



### 2. class별 개수


```python
train_df[['class', 'label']].groupby('class').count().rename(columns={'label':'count'})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>class</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bottle</th>
      <td>241</td>
    </tr>
    <tr>
      <th>cable</th>
      <td>271</td>
    </tr>
    <tr>
      <th>capsule</th>
      <td>275</td>
    </tr>
    <tr>
      <th>carpet</th>
      <td>327</td>
    </tr>
    <tr>
      <th>grid</th>
      <td>294</td>
    </tr>
    <tr>
      <th>hazelnut</th>
      <td>427</td>
    </tr>
    <tr>
      <th>leather</th>
      <td>293</td>
    </tr>
    <tr>
      <th>metal_nut</th>
      <td>268</td>
    </tr>
    <tr>
      <th>pill</th>
      <td>340</td>
    </tr>
    <tr>
      <th>screw</th>
      <td>381</td>
    </tr>
    <tr>
      <th>tile</th>
      <td>273</td>
    </tr>
    <tr>
      <th>toothbrush</th>
      <td>75</td>
    </tr>
    <tr>
      <th>transistor</th>
      <td>233</td>
    </tr>
    <tr>
      <th>wood</th>
      <td>278</td>
    </tr>
    <tr>
      <th>zipper</th>
      <td>301</td>
    </tr>
  </tbody>
</table>
</div>



### 3. state별 개수


```python
train_df[['class', 'state']].groupby('state').count().rename(columns={'class':'count'})
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>state</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bent</th>
      <td>19</td>
    </tr>
    <tr>
      <th>bent_lead</th>
      <td>5</td>
    </tr>
    <tr>
      <th>bent_wire</th>
      <td>7</td>
    </tr>
    <tr>
      <th>broken</th>
      <td>6</td>
    </tr>
    <tr>
      <th>broken_large</th>
      <td>10</td>
    </tr>
    <tr>
      <th>broken_small</th>
      <td>11</td>
    </tr>
    <tr>
      <th>broken_teeth</th>
      <td>10</td>
    </tr>
    <tr>
      <th>cable_swap</th>
      <td>6</td>
    </tr>
    <tr>
      <th>color</th>
      <td>48</td>
    </tr>
    <tr>
      <th>combined</th>
      <td>29</td>
    </tr>
    <tr>
      <th>contamination</th>
      <td>22</td>
    </tr>
    <tr>
      <th>crack</th>
      <td>43</td>
    </tr>
    <tr>
      <th>cut</th>
      <td>28</td>
    </tr>
    <tr>
      <th>cut_inner_insulation</th>
      <td>7</td>
    </tr>
    <tr>
      <th>cut_lead</th>
      <td>5</td>
    </tr>
    <tr>
      <th>cut_outer_insulation</th>
      <td>5</td>
    </tr>
    <tr>
      <th>damaged_case</th>
      <td>5</td>
    </tr>
    <tr>
      <th>defective</th>
      <td>15</td>
    </tr>
    <tr>
      <th>fabric_border</th>
      <td>9</td>
    </tr>
    <tr>
      <th>fabric_interior</th>
      <td>8</td>
    </tr>
    <tr>
      <th>faulty_imprint</th>
      <td>21</td>
    </tr>
    <tr>
      <th>flip</th>
      <td>12</td>
    </tr>
    <tr>
      <th>fold</th>
      <td>9</td>
    </tr>
    <tr>
      <th>glue</th>
      <td>16</td>
    </tr>
    <tr>
      <th>glue_strip</th>
      <td>9</td>
    </tr>
    <tr>
      <th>good</th>
      <td>3629</td>
    </tr>
    <tr>
      <th>gray_stroke</th>
      <td>8</td>
    </tr>
    <tr>
      <th>hole</th>
      <td>23</td>
    </tr>
    <tr>
      <th>liquid</th>
      <td>5</td>
    </tr>
    <tr>
      <th>manipulated_front</th>
      <td>12</td>
    </tr>
    <tr>
      <th>metal_contamination</th>
      <td>15</td>
    </tr>
    <tr>
      <th>misplaced</th>
      <td>5</td>
    </tr>
    <tr>
      <th>missing_cable</th>
      <td>6</td>
    </tr>
    <tr>
      <th>missing_wire</th>
      <td>5</td>
    </tr>
    <tr>
      <th>oil</th>
      <td>9</td>
    </tr>
    <tr>
      <th>pill_type</th>
      <td>5</td>
    </tr>
    <tr>
      <th>poke</th>
      <td>20</td>
    </tr>
    <tr>
      <th>poke_insulation</th>
      <td>5</td>
    </tr>
    <tr>
      <th>print</th>
      <td>9</td>
    </tr>
    <tr>
      <th>rough</th>
      <td>17</td>
    </tr>
    <tr>
      <th>scratch</th>
      <td>47</td>
    </tr>
    <tr>
      <th>scratch_head</th>
      <td>12</td>
    </tr>
    <tr>
      <th>scratch_neck</th>
      <td>13</td>
    </tr>
    <tr>
      <th>split_teeth</th>
      <td>9</td>
    </tr>
    <tr>
      <th>squeeze</th>
      <td>10</td>
    </tr>
    <tr>
      <th>squeezed_teeth</th>
      <td>8</td>
    </tr>
    <tr>
      <th>thread</th>
      <td>16</td>
    </tr>
    <tr>
      <th>thread_side</th>
      <td>12</td>
    </tr>
    <tr>
      <th>thread_top</th>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>


### label별 개수


```python
labelCount = train_df[['class', 'label']].groupby('label').count().rename(columns={'class':'count'})
labelCount
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bottle-broken_large</th>
      <td>10</td>
    </tr>
    <tr>
      <th>bottle-broken_small</th>
      <td>11</td>
    </tr>
    <tr>
      <th>bottle-contamination</th>
      <td>11</td>
    </tr>
    <tr>
      <th>bottle-good</th>
      <td>209</td>
    </tr>
    <tr>
      <th>cable-bent_wire</th>
      <td>7</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>zipper-fabric_interior</th>
      <td>8</td>
    </tr>
    <tr>
      <th>zipper-good</th>
      <td>240</td>
    </tr>
    <tr>
      <th>zipper-rough</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-split_teeth</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-squeezed_teeth</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>88 rows × 1 columns</p>
</div>




```python

```

### class별 label 종류와 개수


```python
anomaly_dict = {}
for className in classList:
    df = pd.DataFrame(labelCount[labelCount.index.str.contains(className)]).sort_values(by='count', ascending=False)
    display(df)
    anomaly_dict[className] = df
```


<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>transistor-good</th>
      <td>213</td>
    </tr>
    <tr>
      <th>transistor-bent_lead</th>
      <td>5</td>
    </tr>
    <tr>
      <th>transistor-cut_lead</th>
      <td>5</td>
    </tr>
    <tr>
      <th>transistor-damaged_case</th>
      <td>5</td>
    </tr>
    <tr>
      <th>transistor-misplaced</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>capsule-good</th>
      <td>219</td>
    </tr>
    <tr>
      <th>capsule-crack</th>
      <td>12</td>
    </tr>
    <tr>
      <th>capsule-scratch</th>
      <td>12</td>
    </tr>
    <tr>
      <th>capsule-faulty_imprint</th>
      <td>11</td>
    </tr>
    <tr>
      <th>capsule-poke</th>
      <td>11</td>
    </tr>
    <tr>
      <th>capsule-squeeze</th>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>wood-good</th>
      <td>247</td>
    </tr>
    <tr>
      <th>wood-scratch</th>
      <td>11</td>
    </tr>
    <tr>
      <th>wood-combined</th>
      <td>6</td>
    </tr>
    <tr>
      <th>wood-hole</th>
      <td>5</td>
    </tr>
    <tr>
      <th>wood-liquid</th>
      <td>5</td>
    </tr>
    <tr>
      <th>wood-color</th>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>bottle-good</th>
      <td>209</td>
    </tr>
    <tr>
      <th>bottle-broken_small</th>
      <td>11</td>
    </tr>
    <tr>
      <th>bottle-contamination</th>
      <td>11</td>
    </tr>
    <tr>
      <th>bottle-broken_large</th>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>screw-good</th>
      <td>320</td>
    </tr>
    <tr>
      <th>screw-scratch_neck</th>
      <td>13</td>
    </tr>
    <tr>
      <th>screw-manipulated_front</th>
      <td>12</td>
    </tr>
    <tr>
      <th>screw-scratch_head</th>
      <td>12</td>
    </tr>
    <tr>
      <th>screw-thread_side</th>
      <td>12</td>
    </tr>
    <tr>
      <th>screw-thread_top</th>
      <td>12</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>cable-good</th>
      <td>224</td>
    </tr>
    <tr>
      <th>cable-bent_wire</th>
      <td>7</td>
    </tr>
    <tr>
      <th>cable-cut_inner_insulation</th>
      <td>7</td>
    </tr>
    <tr>
      <th>cable-cable_swap</th>
      <td>6</td>
    </tr>
    <tr>
      <th>cable-combined</th>
      <td>6</td>
    </tr>
    <tr>
      <th>cable-missing_cable</th>
      <td>6</td>
    </tr>
    <tr>
      <th>cable-cut_outer_insulation</th>
      <td>5</td>
    </tr>
    <tr>
      <th>cable-missing_wire</th>
      <td>5</td>
    </tr>
    <tr>
      <th>cable-poke_insulation</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>carpet-good</th>
      <td>280</td>
    </tr>
    <tr>
      <th>carpet-color</th>
      <td>10</td>
    </tr>
    <tr>
      <th>carpet-thread</th>
      <td>10</td>
    </tr>
    <tr>
      <th>carpet-cut</th>
      <td>9</td>
    </tr>
    <tr>
      <th>carpet-hole</th>
      <td>9</td>
    </tr>
    <tr>
      <th>carpet-metal_contamination</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>hazelnut-good</th>
      <td>391</td>
    </tr>
    <tr>
      <th>hazelnut-crack</th>
      <td>9</td>
    </tr>
    <tr>
      <th>hazelnut-cut</th>
      <td>9</td>
    </tr>
    <tr>
      <th>hazelnut-hole</th>
      <td>9</td>
    </tr>
    <tr>
      <th>hazelnut-print</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>pill-good</th>
      <td>267</td>
    </tr>
    <tr>
      <th>pill-color</th>
      <td>13</td>
    </tr>
    <tr>
      <th>pill-crack</th>
      <td>13</td>
    </tr>
    <tr>
      <th>pill-scratch</th>
      <td>12</td>
    </tr>
    <tr>
      <th>pill-contamination</th>
      <td>11</td>
    </tr>
    <tr>
      <th>pill-faulty_imprint</th>
      <td>10</td>
    </tr>
    <tr>
      <th>pill-combined</th>
      <td>9</td>
    </tr>
    <tr>
      <th>pill-pill_type</th>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>metal_nut-good</th>
      <td>220</td>
    </tr>
    <tr>
      <th>metal_nut-bent</th>
      <td>13</td>
    </tr>
    <tr>
      <th>metal_nut-flip</th>
      <td>12</td>
    </tr>
    <tr>
      <th>metal_nut-scratch</th>
      <td>12</td>
    </tr>
    <tr>
      <th>metal_nut-color</th>
      <td>11</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>zipper-good</th>
      <td>240</td>
    </tr>
    <tr>
      <th>zipper-broken_teeth</th>
      <td>10</td>
    </tr>
    <tr>
      <th>zipper-fabric_border</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-rough</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-split_teeth</th>
      <td>9</td>
    </tr>
    <tr>
      <th>zipper-combined</th>
      <td>8</td>
    </tr>
    <tr>
      <th>zipper-fabric_interior</th>
      <td>8</td>
    </tr>
    <tr>
      <th>zipper-squeezed_teeth</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>leather-good</th>
      <td>245</td>
    </tr>
    <tr>
      <th>leather-color</th>
      <td>10</td>
    </tr>
    <tr>
      <th>leather-cut</th>
      <td>10</td>
    </tr>
    <tr>
      <th>leather-glue</th>
      <td>10</td>
    </tr>
    <tr>
      <th>leather-fold</th>
      <td>9</td>
    </tr>
    <tr>
      <th>leather-poke</th>
      <td>9</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>toothbrush-good</th>
      <td>60</td>
    </tr>
    <tr>
      <th>toothbrush-defective</th>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>tile-good</th>
      <td>230</td>
    </tr>
    <tr>
      <th>tile-crack</th>
      <td>9</td>
    </tr>
    <tr>
      <th>tile-glue_strip</th>
      <td>9</td>
    </tr>
    <tr>
      <th>tile-oil</th>
      <td>9</td>
    </tr>
    <tr>
      <th>tile-gray_stroke</th>
      <td>8</td>
    </tr>
    <tr>
      <th>tile-rough</th>
      <td>8</td>
    </tr>
  </tbody>
</table>
</div>



<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
    </tr>
    <tr>
      <th>label</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>grid-good</th>
      <td>264</td>
    </tr>
    <tr>
      <th>grid-bent</th>
      <td>6</td>
    </tr>
    <tr>
      <th>grid-broken</th>
      <td>6</td>
    </tr>
    <tr>
      <th>grid-glue</th>
      <td>6</td>
    </tr>
    <tr>
      <th>grid-metal_contamination</th>
      <td>6</td>
    </tr>
    <tr>
      <th>grid-thread</th>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



```python
fig, axs = plt.subplots(15, 1, figsize=(15, 15*5))
fig.subplots_adjust(hspace = .3)
axs = axs.ravel()

for i, (className, df) in enumerate(anomaly_dict.items()):
    colors = ['red' for i in range(len(df.index))]
    colors[0] = 'green'
    axs[i].bar(df.index, df.iloc[:, 0], color=colors, alpha=0.5)
    axs[i].set_title(className, fontsize=20)
    for j, value in enumerate(df.iloc[:, 0]):
        axs[i].text(j, 20, df.iloc[:, 0][j], ha='center', fontsize=20)
```


    
![png](output_43_0.png)


## Class별 State 분포
```python
# 데이터를 그룹별로 정렬
df_agg = train_df.groupby(['class', 'state']).size()
g = df_agg.groupby('class', group_keys=False)
df_agg = g.apply(lambda x: x.sort_values(ascending=False)).reset_index(name='cnt')
df_agg['state'] = np.where(df_agg['state'] == 'good', df_agg['class'], df_agg['state'])
```

```python
f, axs = plt.subplots(1,1,figsize=(30,8))

i = 0
x_label = []
cnt_list = []

for c, color in zip(df_agg['class'].value_counts().index, colors):
    tmp = df_agg.loc[df_agg['class']==c]
    axs.bar(range(i, i+tmp.shape[0]), tmp['cnt'], color = color)
  
    i+= df_agg.loc[df_agg['class']==c].shape[0]
    x_label.extend(list(tmp['class'].unique())+ tmp['state'][1:].to_list())
    cnt_list.extend(tmp['cnt'].to_list())

for x, y in zip(list(range(len(df_agg))), cnt_list):
    axs.annotate('%d\n' %(int(y)), xy=(x,y), textcoords='data', ha = 'center', size = 12) 

axs.set_xticks(range(len(x_label)))
axs.set_xticklabels(x_label, rotation = 85, size = 14)
plt.legend(df_agg['class'].unique())
plt.show()
```

![png](output_49_0.png)

- 같은 class 내에서 state 간 편차가 매우 심함

## Image
## Class Image
```python
fig, axs = plt.subplots(3, 5, figsize=(15, 8))

folder_dir = './data/train'

for i, current_class in enumerate(train_df['class'].unique()):
    image = img.imread(folder_dir+"/"+ train_df.loc[train_df['class'] == current_class]['file_name'].sample(1).iloc[0])
    axs[i//5, i%5].imshow(image)
    axs[i//5, i%5].set_title(current_class, fontsize=15)
    axs[i//5, i%5].axis('off')
plt.show()
```

![png](output_56_0.png)

## Class with state image
```python
folder_dir = './data/train'

for current_class in train_df['class'].unique():
    tmp = train_df.loc[train_df['class'] == current_class]
    states = train_df.loc[train_df['class'] == current_class, 'state'].unique()

    fig, axs = plt.subplots(1, len(states), figsize=(20, 8))

    for col_idx, state in enumerate(states):
        img_dir = tmp.loc[tmp['state'] == state, 'file_name'].sample(1).iloc[0]
        image = img.imread(folder_dir + '/' + img_dir)
        axs[col_idx].imshow(image)
        axs[col_idx].set_xticklabels([])
        axs[col_idx].set_yticklabels([])

    for ax, col in zip(axs, states):
        ax.set_title(col, size = 15)

    axs[0].set_ylabel(current_class, rotation=0, fontsize=15, labelpad=40, fontdict=dict(weight='bold'))

    fig.tight_layout()
    # plt.suptitle(mask_state,fontsize=25, y=1.04)
    plt.show()
    print('\n\n')
```
![png](output_61_0.png)
![png](output_61_2.png)
![png](output_61_4.png)
![png](output_61_6.png)
![png](output_61_8.png)
![png](output_61_10.png)
![png](output_61_12.png)
![png](output_61_14.png)
![png](output_61_16.png)
![png](output_61_18.png)
![png](output_61_20.png)
![png](output_61_22.png)
![png](output_61_24.png)
![png](output_61_26.png)
![png](output_61_28.png)
