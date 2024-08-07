import pandas as pd
import streamlit as st

#사이드바 화면
st.sidebar.header("로그인")
user_id = st.sidebar.text_input("아이디 입력", value="",max_chars=15)
user_password = st.sidebar.text_input("패스워드 입력", value="",type="password")

if user_id=='phar' and user_password == "1234" :

  # 데이터셋 불러오기
  df = pd.read_excel('pharmacy_data.xlsx') 

  df.drop(columns=['Unnamed: 0'], inplace=True)
  df.drop(columns=['weekday'], inplace=True)
  df.drop(columns=['ahumidity'], inplace=True)
  df.drop(columns=['SS'], inplace=True)
  df.drop(columns=['SR'], inplace=True)
  df.drop(columns=['atemp'], inplace=True)
  df.drop(columns=['dtemprange'], inplace=True)
  df.drop(columns=['meanwindspeed'], inplace=True)
  df.drop(columns=['maxwindsnd'], inplace=True)
  df.drop(columns=['maxinwindspeed'], inplace=True)
  df.drop(columns=['maxinwindsnd'], inplace=True)

  #datetime 타입으로 바꾸기
  df['date']=pd.to_datetime(df['date'])  #datetime형식으로 바뀜
  df['year']=df['date'].dt.year
  df['month']=df['date'].dt.month
  df['weekday']=df['date'].dt.weekday
  df['day']=df['date'].dt.day
  df.drop(columns=['date'], inplace=True)

  #데이터 나누기
  # 결측치가 없는 데이터만 사용
  X=df.drop(['count'],axis=1)
  Y=df['count']

  # 데이터 나누기
  from sklearn.model_selection import train_test_split
  #교차검증 조건
  X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)

  #피쳐 스케일링 : 범위표준화
  from sklearn.preprocessing import StandardScaler

  sc = StandardScaler()
  X_train  = sc.fit_transform(X_train)
  X_test = sc.transform(X_test)


  ### 랜덤 포레스트
  from sklearn.ensemble import RandomForestRegressor
  from sklearn.model_selection import GridSearchCV #GridSearchCV 에서 CV는 교차검증
  import warnings
  warnings.filterwarnings('ignore')
  import numpy as np

  random_state_val=2000
  rf_model = RandomForestRegressor(random_state=random_state_val)
  rf_params={'random_state':[random_state_val],'n_estimators':[100],'max_depth':[5]}
  gridsearch_rf_model =GridSearchCV(estimator=rf_model,
                                  param_grid=rf_params,
                                  scoring='accuracy',
                                  cv=5)     # 80%로 나누어지 train 데어터를 다시 cv=5 로 나눔

  #그리드서치 실행
  log_train = np.log(y_train)
  gridsearch_rf_model.fit(X_train,log_train)   #학습 시키는 과정

  # 모델 학습
  rf_model.fit(X_train, y_train)
  # 예측값 생성
  y_test = rf_model.predict(X_test)

  #gridsearch에서는 score로 평가할수 없음
  from sklearn.metrics import mean_squared_log_error, r2_score   #metrics 평가지표

  #예측
  preds= gridsearch_rf_model.best_estimator_.predict(X_test) #예측값과 비교

  # y_true = np.exp(log_train)
  y_pred = np.exp(preds)

  MSLE = mean_squared_log_error(y_test,y_pred)
  R2=r2_score(np.log(y_test),preds)

  MSLE_val= MSLE*100
  R2_val= R2*100


  from datetime import date
  today = date.today()

  #메인 화면(오른쪽 화면)
  st.subheader("조제건수 예측합니다.",divider='rainbow')
  st.write("○오류률: ", f'{MSLE_val:.2f}%')
  st.write("○예측률: ", f'{R2_val:.2f}%')
  st.write("")
  st.write("▷10가지 인자를 입력해주세요")
  st.write("-> 입력 중에는 화면이 흐려집니다. 흐려져도 하단 '저장'을 누를때까지 계속 입력하세요")
  naver_link = "https://search.naver.com/search.naver?where=nexearch&sm=top_sly.hst&fbm=0&acr=1&ie=utf8&query=%EB%82%A0%EC%94%A8+%EC%82%AC%EB%8B%B9%EB%8F%99"
  st.markdown(f"->네이버 날씨로 이동:")
  st.markdown(f"{naver_link}")
  val_year = st.number_input("1.년도 입력(예: 2024년->2024)", value=today.year) 
  val_month = st.number_input("2.월 입력(예: 8월->8)", value=today.month)
  val_day=st.number_input("3. 일 입력(예: 12일 -> 12)", value=today.day)
  val_weekday = st.number_input("4.요일 입력(예:월:0,화:1,수:2,목:3,금:4,토:5)", value=today.weekday()) #min_value=0, max_value=5, step=1, format="%d")
  val_SC= st.number_input("5.예상 일조량을 입력(맑음:0,구름조금:25,구름많음:50,흐림:75)", min_value=0, max_value=100, step=1, format="%d")
  val_mintemp= st.number_input("6.예상 최저(▽)온도를 입력")
  al_maxtemp=st.number_input("7.예상 최고(▲)온도를 입력")
  val_rainfall= st.number_input("8.예상 강수량(mm)을 입력")
  val_maxwindspeed= st.number_input("9.예상 최대풍속을 입력")
  val_minhumidity= st.number_input("10.예상 최저습도 입력")
  # st.write("")


  if st.button("저장"):
    new_data_point =[val_minhumidity,val_SC,val_rainfall,val_mintemp,val_maxwindspeed,val_year,val_month,val_weekday]
    # 랜덤 포레스트 모델에 적용하여 예측값 b 계산
    predicted_b = rf_model.predict([new_data_point])

    st.markdown("---")
    st.write("■ 예측 값을 출력합니다.")

    ##예측값 출력
    if val_weekday ==2 :
      st.write(f"▶▶예측 조제건수: {predicted_b[0]}")
      st.write("▶수요일은 가중치 30%적용한 값도 제공합니다.")
      predicted_b=predicted_b[0] * 1.3
      st.write(f"▶▶가중치적용 예측 조제건수: {predicted_b:.2f}")
    elif val_weekday ==3 :
      st.write(f"▶▶예측 조제건수: {predicted_b[0]}")
      st.write("▶목요일은 가중치 -30%적용한 값도 제공합니다.")
      predicted_b=predicted_b[0] * 0.7
      st.write(f"▶▶가중치적용 예측 조제건수: {predicted_b:.2f} ")
    elif val_weekday == 4 :
      st.write(f"▶▶예측 조제건수: {predicted_b[0]}")
      st.write("▶금요일은 가중치 30%적용한 값도 제공합니다.")
      predicted_b=predicted_b[0] * 1.3
      st.write(f"▶▶가중치적용 예측 조제건수: {predicted_b:.2f} ")
    else:
      st.write(f"▶▶예측 조제건수: {predicted_b[0]}")

  ##   streamlit run pharmacy_model_streamlit.py