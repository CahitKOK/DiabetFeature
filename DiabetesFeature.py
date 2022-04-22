#Görev1:

#Notlar:
# 'DiabetesPedigreeFunction',açken 100 tokluk  140 üzeri diyabet
#'DiabetesPedigreeFunction' :
#BloodPressure : Kan basıncı aralığı üret
#Glucose 140 dan buyukse diyabet
#Age değişkeni üret
# 0 değerlerini ortalama veriye eşitleme
###Belirleyici DiabetesPedigreeFunction,BMI,Pregnancies,Glucose ve Age
#Cilt kalınlığı Deri kıvrım kalınlığı 18.5-24.9,25.0-29.9,30 üstü
## Yaş arttıkça risk artıyor. 60dan sonrası yüksek risk
# Kan basıncı 90 orta ve 60 yüksek risk altındakier
#BMI 27 Altı düşük 27-33 orta 34-41 yükek 42 üstü kesin


import numpy as np
import pandas as pd
import seaborn as sns
import missingno as msno
from datetime import date
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
#Adım 1:
df = pd.read_csv("6-feature_engineering/datasets/diabetes.csv")
df.head()
df.describe().T
df.info()

#İnsulinde %75 ve max değer arasında çok fark olduğu için bir baskılama işlemi yapılabilir
#Adım 2:
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    #Burada yaptığım for döngüsü ile değişkenler içinde col ile gezip eğer "0" ile aynı aynı türden
    ##yani kategorik türünde bir değişkense cat_cols içine at diyoruz.
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"] #Bu kısımda ise cat_th değerinden değişken sınıf sayısı
    #olarak dataframe'in değişkenlerinin sınıf sayısı küçükse ve "0" tipine eşit değilse yani kategorik değilse
    ##bunları numerik gözüküp kategorik olanların içine at diyoruz.
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"] #Burada ise cat_th 'den sınıf sayısının fazla olması ama tip olarak
    #kategorik gözükmesine rağmen kardinal yani ölçülemez olan değişkenleri ifade etmektedir.
    cat_cols = cat_cols + num_but_cat # Burada kategorik değişkenlerimizi tekrar oluşturma sebebimiz
    #numerik gözüküp kategorik olan değişkenleride buraya eklemektir.
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    #Burada ise kategorik olan ama cardinal olmayanları seç ve kategorik değişkenlerimin içine at diyorum

    #num_cols

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    #Burada yaptığımız işlemde değişkenlerin içinden tipi kategorik olmayanları numerik değişkenlere atıyoruz
    #Ayrıca
    num_cols = [col for col in num_cols if col not in num_but_cat]
    #numerik ama kategorik olanların haricindekileride buraya numerik oldukları için ekliyoruz.

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
#Adım 3:
cat_cols, num_cols, cat_but_car = grab_col_names(df)


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)|(dataframe[col_name] == 0)].any(axis=None):
        return True
    else:
        return False
for col in num_cols:
    print(col,check_outlier(df,col))
    #DiabetesPedigreeFunction ve Pregnancies haricinde True dönmemsi gerekir.


#Adım 5:

df["Glucose"].describe().T #Glikoz sıfır olamaz
#BMI,Glucose sıfır olamazlar
df["BMI"].describe().T #Kan basıncı sıfır olamaz
df.loc[(df["BMI"]==0),"BMI"] = df["BMI"].mean()

grab_outliers(df, "BloodPressure")
#Sadece burada var ama bizim sonucumuzu etkilemiyor.
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[((dataframe[variable] < low_limit)|(dataframe[variable] == 0)), variable] = dataframe[variable].std()
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit
num_cols_nan_zero = [col for col in num_cols if col not in ["DiabetesPedigreeFunction","Pregnancies"]]
for col in num_cols_nan_zero:
    print(col, check_outlier(df, col))

for col in num_cols_nan_zero:
    replace_with_thresholds(df,col)
    #Hepsi False dönmüş oldu. low limit ile almış olduk

#Adım 6: Eksik gözlem
df.isnull().values.any()
#eksik gözlem görülmemektedir.

#Adım 7: cor analizi

corr = df.corr()

msno.heatmap(corr)
plt.show()

#Cıktı alamadım

#Görev 2:
#Adım 1:
#AYKIRI GÖZLEMLERDE BU İŞLEM YAPILMIŞTI
#ancak yukarıda Pregnancies ile inulin arasında bağlantı bulduğumuz için
##İnulinide değiştirmekte fayda var ortalaması ile değiştiriyoruz
df.describe().T
df["INSULIN"].describe()
df.loc[(df["INSULIN"]<=0),"INSULIN"] = df["INSULIN"].std()
check_outlier(df,"Insulin") #False aldık  
#Adım 2 :Yeni değişkenler
df.head()
df.columns = [col.upper() for col in df.columns]
df.describe().T
# age level
df.loc[(df['AGE'] <30 ), 'NEW_AGE_CAT'] = 'young_lowrisk'
df.loc[(df['AGE'] >= 30) & (df['AGE'] < 60), 'NEW_AGE_CAT'] = 'mature_risk'
df.loc[(df['AGE'] >= 56), 'NEW_AGE_CAT'] = 'senior_highrisk'

#GLUCOSE levet
df.loc[(df['GLUCOSE'] < 140), 'NEW_GLUCOSE_CAT'] = "notrisk"
df.loc[(df['GLUCOSE'] >= 140), 'NEW_GLUCOSE_CAT'] = "highrisk"
# INSULIN x PREGNANCIES
df["PREGNANCIES"].astype(int)
df.loc[(df['INSULIN'] >300) & (df['PREGNANCIES'] <= 1), 'NEW_RİSK_CAT'] = 'lowrisk'
df.loc[(df['INSULIN'] >300) & (df['PREGNANCIES'] > 1)& (df['PREGNANCIES'] <= 3), 'NEW_RİSK_CAT'] = 'risk'
df.loc[(df['INSULIN'] >300) & (df['PREGNANCIES'] > 3), 'NEW_RİSK_CAT'] = 'risk'
df.loc[(df['INSULIN'] <= 300) & (df['INSULIN'] >=150) & (df['PREGNANCIES'] <= 1), 'NEW_RİSK_CAT'] = 'lowrisk'
df.loc[(df['INSULIN'] <= 300) & (df['INSULIN'] >=150) & (df['PREGNANCIES'] > 1), 'NEW_RİSK_CAT'] = 'risk'
df.loc[(df['INSULIN'] < 150 ) & (df['PREGNANCIES'] > 3), 'NEW_RİSK_CAT'] = 'highrisk'
df.loc[(df['INSULIN'] <150) & (df['PREGNANCIES'] > 1)& (df['PREGNANCIES'] <= 3), 'NEW_RİSK_CAT'] = 'highrisk'
df.loc[(df['INSULIN'] < 150 ) & (df['PREGNANCIES'] <= 1), 'NEW_RİSK_CAT'] = 'risk'
#AGE X DIABETESPEDIGREEFUNCTION
df["NEW_AGE_FUNCTİON"] = df["AGE"] * df["DIABETESPEDIGREEFUNCTION"]
#BLOODPRESSURE LEVEL
df.loc[(df['BLOODPRESSURE'] > 90 ), 'NEW_BLOODPRESSURE_CAT'] = 'lowrisk'
df.loc[(df['BLOODPRESSURE'] > 60) & (df['BLOODPRESSURE'] <= 90), 'NEW_BLOODPRESSURE_CAT'] = 'risk'
df.loc[(df['BLOODPRESSURE'] < 60), 'NEW_BLOODPRESSURE_CAT'] = 'highrisk'
#BMI LEVEL
df.loc[(df['BMI'] < 27 ), 'NEW_BMI_CAT'] = 'lowrisk'
df.loc[(df['BMI'] > 27) & (df['BMI'] <= 41), 'NEW_BMI_CAT'] = 'risk'
df.loc[(df['BMI'] > 42), 'NEW_BMI_CAT'] = 'highrisk'
#SKINTHICKNESS LEVEL (18.5-24.9,25.0-29.9,30 üstü)
df.loc[(df['SKINTHICKNESS'] < 18.5 ), 'NEW_SKINTHICKNESS_CAT'] = 'lowrisk'
df.loc[(df['SKINTHICKNESS'] >= 18.5) & (df['SKINTHICKNESS'] <= 24.9), 'NEW_SKINTHICKNESS_CAT'] = 'risk'
df.loc[(df['SKINTHICKNESS'] >= 25.0) & (df['SKINTHICKNESS'] <= 29.9), 'NEW_SKINTHICKNESS_CAT'] = 'highrisk'
df.loc[(df['SKINTHICKNESS'] > 30), 'NEW_SKINTHICKNESS_CAT'] = 'riskkk'
df.info()
df.head()

#Adım 3:
#Label
#----------------------------------------------
# 4. Label Encoding
#----------------------------------------------
def label_encoder(dataframe,binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


#----------------------------------------------
# 5. Rare Encoding
#----------------------------------------------
def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")
rare_analyser(df, "OUTCOME", cat_cols)

def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df, 0.01)

df.head()

#----------------------------------------------
# 6. One-Hot Encoding
#----------------------------------------------
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

df.head()
df.shape
df.info()

cat_cols, num_cols, cat_but_car = grab_col_names(df)


# df.drop(useless_cols, axis=1, inplace=True)

#----------------------------------------------
# 7. Standart Scaler
#----------------------------------------------

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

df.head()
df.shape



#############################################
# 8. Model
#############################################

y = df["OUTCOME"] #bağımlı değişken
X = df.drop(["OUTCOME"], axis=1) #bağımsız değişkenler bunların haricindekiler

#Değişkenlerimizi belirliyoruz.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,random_state=17)
#Bu değişkenleri train ve test olarak ik farklı sınıfa ayırıyoruz. Trainler ile model kurup test ile bunları
##denetliyor olacağız

from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=46).fit(X_train,y_train)
y_pred = rf_model.predict(X_test)
#Burada önce modelin x_testini tahmin etmesini istiyoruz.
##Daha sonra biz bu tahmin değerleri ile elimizdeki değerlerin karşılaştırılmasını yaparak
### skorumuzu alıyoruz.
accuracy_score(y_pred,y_test)
#Doğruluk skorunu test edip sonucu almak için ilk modelin bulduğunu ikinci olarak

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                      ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X_train)
