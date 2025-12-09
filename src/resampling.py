from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.combine import SMOTEENN

def apply_smote(X, y, random_state=42):
    sm = SMOTE(random_state=random_state)
    X_res, y_res = sm.fit_resample(X, y)
    return X_res, y_res

def apply_adasyn(X, y, random_state=42):
    ad = ADASYN(random_state=random_state)
    X_res, y_res = ad.fit_resample(X, y)
    return X_res, y_res

def apply_random_oversample(X, y, random_state=42):
    over = RandomOverSampler(random_state=random_state)
    X_res, y_res = over.fit_resample(X, y)
    return X_res, y_res

def apply_random_undersample(X, y, random_state=42):
    under = RandomUnderSampler(random_state=random_state)
    X_res, y_res = under.fit_resample(X, y)
    return X_res, y_res

def apply_smoteenn(X, y):
    sampler = SMOTEENN(random_state=42)
    X_res, y_res = sampler.fit_resample(X,y)
    return X_res, y_res
    
def resplit_and_scale(X, y, random_state=42, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test
