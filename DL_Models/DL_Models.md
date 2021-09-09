## CNN Convolutional Neural Network

![CNN](Image\CNN.png)

CNN은 **Convolutions** 과 **Subsampling** 레이어로 구성이 되어 **Convolutions, Subsampling** 레이어가 반복 되다가 마지막에

**Fully connected layers**가 나오게 됩니다.



CNN은 **Convolutions** 와 **Subsampling** 레이어가 **Feature extraction**을 하게 됩니다.

여기서 **Feature extraction** 는 개와 고양이를 구분을 할때 개와 고양이의 귀 그리고 꼬리 같은 이미지의 조합으로 이미지를 구분을

하게 되는데 여기서 중요한  특징이나 패턴 같은 걸 뽑아내는 것을 **Feature extraction** 이라고 합니다.



그리고 **Feature extraction** 한 정보를 바탕으로 개와 고양이를 구분을 하는 것이 바로 **Classifier** 입니다. 그리고 그 것들이 바로 

**Fully connected layers** 입니다.



즉 중요한 정보들을 **Convolutions** 레이어와 **Subsampling** 레이어가 얻어내고 그 것들을 바탕으로 최종적으로 원하는 분류를 하는데

**Fully connected layers**가 필요합니다.



![CNN_2](Image\CNN_2.png)

여기서 **Convolutions** 와 **Subsampling ( Pooling 은 Subsampling의 일종 )** **Feature extraction** 을 한 후에 **Fully connected layers**

로 최종 **Output**을 내게 됩니다. 여기서 최종 **Output**은 **boat** 입니다.



### CNN이 강력한 이유

* **Local Invariance**

  **Convolution filters** 가 전체 이미지를 **모두** 돌아다니며 검사를 하기 때문에 찾고 싶은 물체가 이미지 안에 어디에 위치하는지는

  중요해지지 않습니다.

* **Compositionality**

  이미지가 주어진다면 거기에 Convolution이 생기고 **CNN**이 돌아가서 거기 위에 또 Convolution 이루어 지고 이런 식으로 계층 구조

  를 쌓게 되는데 이런 것을 일종의 **Compositionality** 라고 합니다.

![CNN_3](Image\CNN_3.png)

위 그림 처럼 필터가 돌아가면서 Convolution을 하게 됩니다. 여기서 필터의 모양은 학습을 통해서 최적의 필터 모양을 찾습니다.

* Zero-Padding

![CNN_4](Image\CNN_4.png)

​	위 그림에선 1*5 사이즈의 이미지를 Convolution을 합니다.

​	여기서 2, 3, 4번째는 Convolution이 잘 되지만, 첫 번째나 다섯 번째 즉 가장자리 부분은 Convolution을 할 수가 없습니다.

​	따라서 가장자리에 0을 채워서 Convolution이 가능하게 한게 Zero-Padding 입니다.

​	위 그림에서 input의 크기는 5, output의 크기는 5, filter의 크기는 3 이기 때문에 Zero-padding의 크기는 1 입니다.

​	이것을 수식으로 정리를 해보자면 아래와 같습니다.
$$
n_{out} = (n_{in} + 2*n_{padding} - n_{filter}) + 1
$$

* Stride

  ![CNN_5](Image\CNN_5.png)

  같은 Convolution Filter 모양이 해당 픽셀을 매 칸마다 Convolution을 할 수도 있고, 오른쪽 그림 처럼 두 칸 마다 Convolution를

  할 수 있습니다. 따라서 왼쪽 그림의 Stride 는 1 이고, 오른쪽 그림의 Stride 는 2 입니다.

  그리고 또한 Stride size 와 Filter size가 같다면 overlapping 을 방지 할 수 있습니다.

## AlexNet

![image-20210901192338298](Image\CNN_6.png)

AlexNet 2012년에 개최된 ILSVRC(ImageNet Large Scale Visual Recognition Challenge) 대회의 우승을 차지한 CNN의 일종입니다.



ILSVRC는 컴퓨터 비전에서 객체 인식, 즉 현실적인 이미지 데이터에서, 어떤 특정한 객체가 어디에 위치하고, 어떤 종류인지를 알아내는 대회로, 정확도, 속도 등의 기준으로 우열을 가리는 대회입니다.

이 객체 인식 대회에서는 여러 알고리즘과 기술들이 사용되었고 딥러닝은 이론만 장황하고 성능은 형편없는 기술로 여겨졌었습니다. 당시 가장 대세였고 성능이 좋던 SVM + HoG 모델을 제치고, 딥러닝 기술이 컴퓨터 비전에서 최고 성능을 낼 수 있다고 증명을 했습니다.



AlexNet이랑 거의 비슷한 LeNet이 이미 1998년도에 개발이 되었지만, 하드웨어 성능 상승과 GPU 병렬처리로 인해 성능을 두각을 내지 못했었습니다.



AlexNet이 LeNet의 개념을 다듬고, 병렬 처리 및, 활성화 함수를 바꿈으로 연산 속도를 빠르게 만든 것이 가장 큰 발전입니다.



위의 그림에서 필터가 2개로 나와있는데 이는 병렬연산을 위한 분할입니다.



AlexNet의 특징

* 활성화 함수를 ReLU를 사용합니다.

* 기존 까지는 Sigmoid를 필두로 tanh를 사용하는 추세 였는데,

  AlexNet이 ReLU를 사용했을때 학습, 예측 속도가 빠르고, 정확도는 유지되었습니다.

  이후 ReLU가 자주 사용되며, Leaky ReLU, ELU등 관련된 활성화 함수를 사용합니다.

fully connected layer에선 Overfitting을 막기 위해 Dropout을 사용하고, 오버레핑 풀링을 시행한 결과 정확도가 올랐다고 합니다.

* Data agumentation

  ![image-20210901200059915](Image\CNN_7.png)

  ​	Data augmentation 은 말 그대로 기존 데이터의 약간 수정된 사본 또는 기존 데이터에서 새로 생성 된 합성 데이터를 추가하여

  ​	데이터 양을 늘리는데 사용되는 기술입니다.

  ![image-20210901200302483](Image\CNN_8.png)

  Data augmentation 을 통해 (256 - 224) * (256 - 224) * 2 를 해서 데이터를 2048배 증식 시킬 수 있습니다.

  ![image-20210901200610696](Image\CNN_9.png)

  또한 Color variation 를 통해 RGB 값을 변경하여 데이터를 증식 시킬 수 있습니다.

  하지만 너무 많이 RGB 값을 변경을 하게 된다면 전혀 다른 Label 값이 될 수도 있기 때문에, 모델이 허용할 수 있는 범위만큼의

  노이즈를 줘야 합니다.





## VGG

![image-20210901200906211](Image\CNN_10.png)

 VGG는 AlexNet과 많은 차이는 없습니다. 단 8 레이어 였던 것이 16 레이어로 늘어나고, 3x3의 작은 필터를 사용해서 stride를 1로 주고,

stride를 1로 주고, padding을 1로 주고, 2x2의 윈도우로 맥스풀링을 stride 2단위로 진행을 했습니다.ㄴ



필터가 검사하는 특징 부분이 세밀해지고, 레이어도 깊어짐으로써 보다 다양한 형태를 파악 할 수 있기에 정확도가 올라갑니다.



연산량도 역시 올라가지만, GPU 병렬처리로 많은 연산량은 어느정도 해소가 가능하기에 별로 문제가 되지 않습니다.

VGG16와 VGG19가 존재하는데, 뒤의 숫자는 레이어의 갯수를 의미합니다. 따라서 VGG19가 성능이 더 좋습니다.



## GoogLeNet

GoogLeNet은 2014년 ILSVRC에서 VGGNet(VGG19)를 이기고 우승을 차지한 알고리즘입니다. GoogLeNet은 19층의 VGG19보다 더 깊은

22층으로 구성되어 있습니다. GoogLeNet 라는 이름에서 알 수 있듯이 구글이 개발에 참여한 알고리즘 입니다.

![image-20210901202431339](Image\CNN_11.png)

GoogLeNet은 22개 층으로 구성이 되어 있습니다.

* 1 x 1 컨볼루션

![image-20210901202522125](Image\CNN_12.png)

​	GoogLeNet은 특이하게 1 x 1 사이즈의 필터로 컨볼루션을 해줍니다. 1 x 1 컨볼루션은 파라미터의 갯수를 줄이는 용도로 

​	사용이 됩니다. 파라미터의 갯수가 줄어들면 그만큼 연산량이 줄어듭니다. 연산량을 줄인다는 점은 네트워크를 더 깊이

​	만들 수 있게 도와준다는 점에서 중요합니다. 

![image-20210901222853385](Image\CNN_13.png)

​	1 x 1 컨볼루션이 없으면 112.9M의 파라미터를 가지고 1 x 1 컨볼루션이 있으면 5.3M 으로 112.9M에 비해 매우 적은

​	파라미터를 가지고 있는 것을 알 수 있습니다.



* Inveption 모듈

![image-20210901223135340](Image\CNN_14.png)

​	GoogLeNet은 총 9개의 인셉션 모듈을 포함하고 있습니다.

![image-20210901223234949](Image\CNN_15.png)

​	GoogLeNet에 실제로 사용된 모델은 1 x 1 컨볼루션이 포함된 b 모델 입니다. 노란색 블럭으로 표현된 1 x 1 컨볼루션을 제외한

​	naive 버전을 보면 이전 층에서 생성된 1 x 1 컨볼루션, 3 x 3 컨볼루션, 5 x 5 컨볼루션, 3 x 3 max pooling 을 해준 결과 얻은

​	파라미터들을 모두 함게 쌓아줍니다. AlexNet, VGGNet 처럼 이전 CNN 모델은 한 층에서 동일한 사이즈의 필터커널을 이용해서

​	컨볼루션을 해줬던 것과 차이가 있습니다. 따라서 다양한 종류의 특성이 도출 됩니다.

* global average pooling

  AlexNet, VGGNet 등에서는 fully connected 층들이 망의 후반부에 있었습니다. 그러나 GoogLeNet은 FC 방식 대신에

  global average pooling이라는 방식을 사용합니다. global average pooling은 전 층에서 뽑아낸 파라미터들을 각각 평균낸 것을

  이어서 1차원 벡터를 만들어주는 것 입니다. 1차원 벡터를 만들어줘야 최종적으로 이미지 분류를 위한 Softmax 층을 연결해줄 수

  있습니다. 만약 전 층에서 1024장의 8 x 8 특성맵이 생성되었다면, 1024장의 8 x 8 특성맵 각각 평균내주어 얻은 1024개의 값을

  하나의 벡터로 연결해주는 것 입니다.

  ![image-20210901230004592](Image\CNN_16.png)

  이 단계로 가중치의 갯수를 많이 제거 할 수 있습니다. 만약 FC 방식을 활용한다면 8 * 8 * 1024 * 1024 = 67.1M 이지만

  Global Average Pooling을 사용한다면 가중치가 필요하지 않습니다.



* #### auxiliary classifier

  네트워크의 깊이가 깊어지면 깊어질수록 기울기 소실(vanishing gradient) 문제를 피하기 어려워집니다. 그러니까 가중치를

  훈련하는 과정에서 역전파(back propagation)를 주로 활용합니다, 하지만 역전파과정에서 가중치를 업데이트하는데 사용되는

  gradient가 점점 작아져서 0이 되어 버립니다. 따라서 네트워크 내의 가중치들이 제대로 훈련되지 않습니다. 이 문제를 극복하기

  위해 GoogLeNet에서는 네트워크 중간에 두 개의 보조 분류기(auxiliary clasifier)를 달아주었습니다.

  
  
  ![image-20210901232113849](Image\CNN_17.png)

## ResNet

ResNet은 2015년 ILSVRC에서 우승을 차지한 CNN 모델 입니다. ResNet은 마이크로소프트에서 개발한 알고리즘 입니다.

ResNet은 2014년의 GoogLeNet이 22개의 층으로 구성된 것에 비해, ResNet은 152개의 층을 갖습니다.

![image-20210901230004592](Image\CNN_18.png)

위 그림처럼 네트워크가 깊어지면서 top-5 error가 낮아진 것을 확인 할 수 있습니다. 

![image-20210901230004592](Image\CNN_19.png)

하지만 더 깊은 구조를 갖는 56층의 네트워크가 20층의 네트워크 보다 더 나쁜 성능을 보입니다.

따라서 망을 무조건 깊게한다고 성능이 늘어나는 것이 아닙니다. 따라서 새로운 방법이 있어야 망을 깊게 만드는 효과를 볼 수 있습니다.



**Residual Block**

![image-20210901230004592](Image\CNN_20.png)

* 그림에서 오른쪽이 Residual Block을 나타낸다. 기존의 망과 차이는 입력값을 출력값에 더해줄 수 있도록 지름길을 하나 만들어 준 것입니다.

* 기존의 신경망은 입력값 X를 타겟값 Y로 매핑하는 함수 H(x)를 얻는 것이 목적이였습니다. 그러나 ResNet은 F(x) + x를 최소화하는

  것을 목적으로 합니다. x는 변할 수 없는 값이므로 F(x)를 0에 가깝게 만드는 것이 목적입니다. F(x)가 0이 되면 출력과 입력이 모두 x로

  같아지게 됩니다. F(x) = H(x) - x 이므로 F(x)를 최소로 해준다는 것은 H(x) - x를 최소로 해주는 것과 동일한 의미를 지닙니다.

  여기서 H(x) - x 를 잔차(residual)이라고 합니다. 즉 잔차를 최소를 해주는 것이므로 ResNet이란 이름이 붙게 됩니다.

**ResNet의 구조**

ResNet은 기본적으로 VGG-19의 구조를 뼈대로 합니다. 거기에 컨볼루션 층들을 추가해서 깊게 만든 후에, shortcut들을 추가합니다.

아래 사진은 34층의 ResNet과 거기에서 shortcut들을 제외한 버전인 plain 네트워크 구조 입니다.

![image-20210901230004592](Image\CNN_21.png)

그럼처럼 34층의 ResNet은 처음을 제외하고 균일하게 3 x 3 사이즈의 컨볼루션 필터를 사용했습니다. 그리고 파라미터의 사이즈가

반으로 줄어들 때 특성맵의 뎁스를 2배로 높혔습니다.



아래 사진은 이미지넷에서 18층 및 34층의 plain 네트워크와 ResNet의 성능을 비교한 사진 입니다.

![image-20210901230004592](Image\CNN_22.png)

왼쪽 그래프를 보면 plain 네트워크는 망이 깊어지면서 오히려 에러가 커졌습니다. 34층의 plain 네트워크가 18층의 plain 네트워크보다 성능이 나쁩니다. 반면, 오른쪽 그래프의 ResNet은 망이 깊어지면서 에러도 역시 작아졌습니다. 보시다 싶이 shortcut을 연결해서 잔차(residual)를 최소가 되게 학습한 효과가 있습니다.



아래의 사진은 18, 34, 50, 101, 152층의 성능과 구성을 나타낸 사진 입니다. 깊은 구조일수록 성능이 좋습니다. 따라서 152층의 ResNet이 가장 성능이 뛰어납니다.

![image-20210901230004592](Image\CNN_23.png)

## regularization

특정 데이터셋에 대한 학습을 진행할 때, 그 데이터에만 모델이 너무 잘 맞춰져서 일반적인 데이터에는 성능이 오히려 좋지 못하게 되는

경우를 Overfitting(과적합)이라고 합니다. 보통 모델을 학습할 때, training data와 test data로 나뉘는데 학습은 training data를 기준으로 

하지만 모델이 overfitting 됐는지 검증하기 위해서 training data와는 다른, 비슷한 일반적인 데이터인 test data를 사용합니다.



**정규화(Regularization)이란, Overfitting을 막고 모델의 일반화를 위해 적용하는 것 입니다.**

1. Dropout

   ![image-20210901230004592](Image\Dropout.png)

   Dropout이란, 위 그림에서 왼쪽 그림과 같은 모델에서 몇개의 연결을 끊어서, 즉 몇개의 노드를 죽이고 남은 노드들을 통해서만 훈련을 하는 것입니다. 이때 죽이는, 쉬게하는 노드들을 랜덤하게 선택합니다.

2. Early Stopping

   Early Stopping이란, 모델이 데이터에 과적합되기 전에 적당한 시기에 학습을 멈추는 것 입니다.

   test data에 대한 성능이 최저가 될 때 멈출 수 있습니다.

   ![image-20210901230004592](Image\Early.png)

3. Batch Normalization

   ![image-20210901230004592](Image\Batch_N_1.png)

   Batch Normalization이란, Layer의 출력 분포가 한쪽으로 치우쳐있는 경우, 학습 성능이 떨어지는데 이를 normal gaussian 분포로 바꾸어주는 레이어 입니다.

   ![image-20210901230004592](Image\Batch_N_2.png)

   데이터의 분포의 평균값과 분산을 구하여 normal gaussian (0, 1)의 분포로 간단히 변환시켜주는 것입니다.



​	일반적으로 Dropout과 Batch normalization은 학습 성능을 개선하고, Dropout은 학습 속도도 높힌다고 합니다.



4. Parameter Norm Penalties

   ![image-20210903140422178](Image\Parameter_Norm_Penalties.png)

   비용함수에 제곱을 더하거나(L2) 절댓값을 더해서 (L1) 가중치의 크기에 제한을 줍니다.

   * L2 weight decay (제곱값)

   * L1 weight decay (절댓값)

     

5. Dataset Augmentation

   머신러닝에서 가장 효과적인 정규화 방법은 학습셋의 크기를 늘리는 것 입니다. 물론 우리가 가지고 있는 데이터의 수는 제한적이기 

   때문에, 학습셋에 가짜 데이터(fake data)를 포함할 수 있습니다.

   Augmentation 하는 방법

   * 이미지 반전
   * 이미지 밝기 조절
   * 서브 샘플링 (Subsampling)
   * 노이즈(noise) 넣기

   학습셋의 크기를 늘리기 위해 데이터를 변한할 때, 데이터의 특징을 고려해야 합니다.

   예를 들어, b와 d, 6 과 9 처럼 180도를 뒤집은 것과 같은 데이터의 경우는 좌우 반전하여 데이터를 늘리는 것이 적절하지 않은 방법

   입니다.
   
   
   
6. Noise Robustnes

   ![image-20210903165541204](Image\Noise_Robustnes.png)

   머신러닝에서 Robust란?

   머신러닝에서 일반화(generalization)는 일부 특성 데이터만 잘 설명하는(overfitting) 것이 아니라 범용적인 데이터도 적합한 모델을

   의미합니다. 즉, 잘 일반화하기 위해서는 이상치나 노이즈가 들어와도 크게 흔들리지 않아야 (robust) 합니다.

   

   Robust한 모델을 만드는 방법

   * 노이즈나 이상치(outlier) 같은 엉뚱한 데이터가 들어와도 흔들리지 않는 모델을 만들기 위한 방법으로 일부러 노이즈를 주는 방법이 있습니다.

   * 레이어 중간에 노이즈를 추가(noise injection)하는 게 파라미터를 줄이는 것(L2 weight decay)보다 강력할 수 있습니다.

     * classification 할 경우 라벨을 부드럽게(label-smoothing) 합니다. (Ex : (1, 0, 0) -> (0.8, 0.1, 0.1))

       

7. Semi-Supervised Learning

   Semi-Supervised Learning는 비지도 학습과 지도 학습을 합친 것으로, 딥러닝에서는 representation을 찾는 과정입니다. CNN으로

   생각해보면, 컨볼루션과 서브샘플링이 반복되는 과정인 특정선택(feature extraction)이 일종의 representation을 찾는 과정입니다.

   컨볼루션 레이어를 정의할 때 사전학습(pre-training)을 하면 비지도 학습에 적합한 (unLabeled) representation을 합니다.

   

8. Multi-Task Learning

   Multi-Task Learning은 한번에 여러 문제를 푸는 모델입니다.

   ![image-20210903190153108](Image\Multi-Task_Learning.png)

   Shared 구조를 덕분에 representation를 잘 찾아줍니다. 서로 다른 문제 (task) 속에서 몇 가지 공통된 중요한 요인 (factor)이 뽑히며,

   shared 구조를 통해서 representation을 찾을 수 있다. 모델의 앞단에 있는 shared 구조 덕분에, 각각의 요인을 학습시킬 때보다

   더 좋은 성능을 냅니다.

   

   딥러닝 관점에서 multi-task learning을 하기 위해서는 모델의 학습셋으로 사용되는 변수는 연관된 다른 모델의 변수와 두 개 이상

   공유한다는 가정이 전제되어야 합니다.

   

9. Parameter Typing and Parameter Sharing

   여러 파라미터(예: CNN)가 있을 때 몇 개의 파라미터를 공유하는 역할을 합니다. 어느 레이어의 파라미터를 공유하거나, 웨이트를 비슷하게 함으로써 각각의 네트워크에 파라미터 수를 줄어드는 효과가 있습니다. 파라미터 수가 줄어들면, 일반적인 퍼포먼스가 증가하는 효과가 있어서 모델의 성능이 좋아지는 데 도움이 됩니다.

   - Parameter Typing (파라미터 수를 줄이는 역할)
     : 입력이 다른데 비슷한 작업(task)을 하는 경우 (예: MNIST, SVHN 데이터셋) 특정 레이어를 공유하거나 두 개의 웨이트를 비슷하게 만듭니다.
   - Parameter Sharing
     : 같은 컨볼루션 필터가 전체 이미지를 모두 돌아다니면서 찍기 때문에 대표적인 예를 CNN이 있습니다.

10. Sparse Representations

    어떤 아웃풋의 대부분 0이 되길 원하는 것 입니다. 히든 레이어가 나오면 히든 레이어의 값을 줄이는 패널티를 추가하면 일종의 Sparse representation 찾는 데 도움이 될 수 있습니다. 

    - Sparse weights (L1 decay)
      : 앞단의 행렬(네트워크의 웨이트)에 0이 많은 것

    - Sparse activations 
      : 뒷단의 행렬에 0이 많은 것으로 더 중요하게 여깁니다

    - - ReLU

      - - 0 보다 작은 activation은 0으로 바꿉니다

        - 아웃풋에 0이 많아지면 sparse activation 할 수 있으므로, 성능이 좋아집니다.

          

11. Bagging and other Ensemble Methods

    앙상블 방법이라고 알려진 model averaging이 정규화에 효과적인 이유는 각각의 모델들은 학습셋에서 같은 오류를 만들지 않기 때문이다.

    - Variance: 어떤 예측을 할 때 결과가 다양하게 나오는 것 입니다.

    - Bias: 평균에서 멀어졌으므로, 그냥 틀린 것 입니다.

    - Bagging: 나온 결과값들을 평균을 냅니다.

      전체 학습셋의 일부를 학습시켜 여러 개의 모델을 만든 뒤, 그 모델에서 도출된 결과값의 평균을 도출하는 것 입니다.

    - Boosting: sequential하게 weak learner 결과값의 차이를 계산하여 학습 모델을 하나씩 더해가는 과정입니다. (AdaBoost)

    

12. Adversrial Training.png

    ![image-20210903190153108](Image\Adversrial Training.png)

    사람이 관측할 수 없을 정도의 작은 노이즈를 넣으면, 완전 다른 클래스가 나옵니다. 입력은 아주 조금 바꼈으나, 출력이 매우 달라지며, 그때의 기울기가 매우 가파릅니다. (파라미터가 조금 변해도 결과값이 달라집니다, overfitting) 오버피팅인데도 성능이 좋은 이유는 엄청나게 많은 모델에 오버피팅을 시켜서 성능이 잘 나옵니다.



## AlphaGo



### 개요

​	알파고란 구글 딥마인드에서 개발한 바둑 인공지능 프로그램입니다. 프로기사를 맞바둑으로 이긴 최초의 프로그램이자 등장과 동시에

​	바둑의 패러다임을 완전히 바꿔버린 인공지능이기도 합니다.



### 바둑이 체스보다 어려운 이유

​	체스, 체커, tic tac toe 등의 게임들은 이미 오래전에 컴퓨터가 인간을 상대로 이긴 분야입니다. 하지만 바둑은 아직까지 인간을

​	뛰어넘지 못했을까요? 그 이유는 바둑에서 발생할 수 있는 경우의 수가 너무나도 많기 때문입니다. 체스, 바둑 등의 턴 방식의 게임들은

​	현 상황에서 앞으로 발생할 수 있는 모든 경우의 수에 대한 탐색을 하여 게임을 진행하게 됩니다. 착수 가능한 수가 평균 250개, 평균

​	게임의 길이가 150수 이기 때문에, 250의 150승 ~ 10의 360이 됩니다. 모든 우주의 원자 보다 훨씬 많은 대단한 숫자입니다.

​	

​	체스는 20년 전의 슈퍼 컴퓨터로도 모든 경우의 수를 탐색이 가능한 크기 였고 지금은 개인 PC에서도 모든 경우의 수를 탐색하는 것이

​	가능하지만, 바둑은 그거와는 비교 할 수 없을 정도로 엄청나게 많은 경우의 수를 보여줍니다. 그렇기 때문에 AI 에서는 바둑을 정복하는

​	것이 최대 과제 중 하나였습니다.



### Monte-Carlo Tree search

바둑의 경우의 수가 원자보다 많이 때문에 brute force 방식은 불가능합니다. 정확한 검색(Exact search)이 힘들기 때문에 

scarch space를 줄이는 적절한 근사 알고리즘 (approximation algorithm)를 사용해야 합니다.



바둑의 search space는 무식하게 큰 숫자 이기때문에, search space를 줄여야 합니다.

따라서 tree의 breadth search를 줄이는 방법, tree의 depth를 줄이는 방법 이 두가지가 필요합니다. 기존의 state-of-art 바둑 시스템들은 

이 방법을 해결하기 위해 Monte-Carlo Tree search (MCTS)라는 방법을 사용하고 있었습니다. 이 방법은 tree search를 정확하게 하는

대신, random하게 node를 하나 고르고 그것을 통해 확률적인 방법으로 대략적인 탐색을 하는 방법입니다.

MCTS는 4단계의 과정을 반복합니다.

![image-20210907214808345](Image\MCTS_1.png)

 	1. Selection : root node에서 부터 Tree Policy를 반복되게 적용해서 leaf node L 까지 도달 한 후에 L을 선택합니다.
 	2. Expansion : 만약 도달한 leaf L에서 게임이 끝나지 않았다면 Tree Policy에 의해 새로운 child node를 만들어 tree 확대합니다.
 	3. Simulation : 새 node 에서 Default Policy 에 따른 결과를 계산합니다.
 	4. Backpropagation : Simulation 결과를 사용해 selection에서 사용하는 통계들을 업데이트 합니다.

* Tree Policy : 이미 존재하는 search tree에서 leaf node를 선택하거나 만드는 Policy 입니다.
* Default Policy : 주어진 non-terminal state에서의 value를 estimation 하는 Polict 입니다.

Backpropagation step는 둘 중 어떤 Policy도 사용하지 않지만, backpropagation을 통해 각 policy들의 parameter들이 update 됩니다.

MCTS는 시간이 허락하는 한도 내에서 이 과정을 계속 반복하고, 그 중에서 가장 좋은 결과를 자신의 다음 action으로 삼습니다.



알고리즘이 강화 학습스러운 방식을 취하기 때문에 휴리스틱적으로 이유가 있고 합리적인 tree를 만들 수 있고, 비대칭 적으로 더 관련이

있는 부분만 집중적으로 Search를 할 수 있습니다.

![11](Image\MCTS_2.png)

### AlphaGo의 접근 방식

​	MCTS가 full search를 하지 않아도 되지만, 바둑에 적용하기 위해서는 breadth와 depth를 줄이는 과정이 필요합니다.

​	AlphaGo에서 이 둘을 줄이기 위하여 사용된 것이 바로 deep learning technique로, 착수하는 지점을 평가하기 위한

​	value network, 그리고 sampling를 하기 위한 distribution를 만들기 위한 Policy network 두 가지 network를 사용하게 됩니다.

![image-20210907220829817](Image\AlphaGo_1.png)

 	1. Selection : 현재 상태에서 Q + u가 가장 큰 지점을 고릅니다.
 	 * Q : MCTS의 action-value 값, 클 수록 승리 확률이 높아집니다.
 	 * u : Policy Network과 node 방문 횟수 등에 의해 결정되는 값 입니다.
 	2. Expansion : 방문 횟수가 40회가 넘으면 child를 expand합니다.
 	3. Evaluation : Value network와 Fast rollout 이라는 두 가지 방법을 사용해 reward를 계산합니다.
 	4. Backup : 시작 지점부터 마지막 leaf node 까지 모든 edge의 parameter를 갱신합니다.
 	5. 1 ~ 4를 시간의 한도 내에서 계속 반복하다가, 가장 많이 방문한 node를 선택합니다.

### Supervised Learning of Policy Network (SL Policy Network)

​	이 네트워크는 CNN으로 구성이 되어 있습니다. Input은 시간이 t일 때의 기보이고, Output은 시간이 t + 1 일 때의 기보(a)가 됩니다.

​	따라서 이 네트워크는 classification network가 됩니다. 이 네트워크는 단순 classification task만 하기 때문에 연속적일 필요는 

​	없습니다. 그래서 모든 (s, a) pair에서 랜덤하게 데이터를 샘플해서 SGD로 학습을 하게 됩니다.



​	네트워크는 총 13 layer CNN을 사용했고, KGS라는 곳에서 3천만 건의 기보 데이터를 가져와 학습을 했습니다. inner product layer은

​	하나도 없이 오직 convolution layer를 학습했습니다.



​	AlphaGo는 이 부분에서 기존 state-of-art였던 44.4% 보다 훨씬 좋은 classification accuracy인 57% 까지 성능개선을 보였습니다.

​	또한 accuracy가 높아지면 높아질수록 최종 winning rate가 상승합니다.

![image-20210907222003971](Image\AlphaGo_2.png)

### Reinforcement Learning of Policy Networks (RL Policy Network)

RL network는 SL network와 동일한 구조를 가지고 있습니다. 초기 값 ρ 역시 SL network의 parameter value σ로 초기화됩니다.

RL network는 현재 RL network policy 와 이전 iteration에서 사용했던 policy network 중에서 랜덤하게 하나를 뽑은 다음 이 둘끼리

서로 대국을 하게 한 후, 둘 중에서 현재 네트워크가 최종적으로 이기면 reward를 +1, 지면 -1을 주도록 디자인이 되어있습니다.

이 네트워크 역시 Stochasic gradient method를 사용해 expected reward를 maximize하는 방식으로 학습이 됩니다.

여기서 일반화된 모델을 만들고 Overfitting을 피하기 위해 과거에 학습된 네트워크를 사용합니다.



SL policy network와 RL policy network가 경쟁할 경우, 거의 80% 이상의 게임을 RL network가 승리했다고 합니다.

 또한 다른 state-of-art 프로그램들과 붙었을 때도 훨씬 좋은 성능을 발휘했다고 합니다.



### Reinforcement Learning of Value Networks

​	Value network는 evaluation 단계에서 사용하는 네트워크로, position s와 policy p가 주어졌을 때, value function 를 predict 하는 

​	네트워크입니다. 

![image-20210907223005687](Image\AlphaGo_3.png)

​	하지만 바둑에 최적의 수를 모르기 때문에 AlphaGo는 가장 우수한 policy인 RL policy network를 사용해 optimal value function을

​	approximation합니다. 

​	Value network는 policy network와 비슷한 구조를 띄고 있지만, 마지막 output layer으로 모든 기보가 아닌 single probability 

​	distribution을 사용합니다.



​	또한 Overfitting 문제를 해결하기 위해 3천만개의 데이터를 RL policy network들끼리의 자가대국을 통해 만들어낸 다음 

​	그 결과를 다시 또 value network를 learning하는 데에 사용합니다. 그래서 training error 0.19, test error 0.37로 overfitting

​	되었던 네트워크가 training error 0.226, test error 0.234로 훨씬 더 일반화 된 네트워크로 학습이 되었습니다.



![image-20210907223609155](Image\AlphaGo_4.png)

​	RL policy를 사용하는 것이 훨씬 높은 우수한 결과를 내는것을 볼 수 있습니다.

## Optimization

### 	Gradient descent

#### 		Batch gradient descent

​			BGD 전체 데이터 셋에 대한 에러를 구한 뒤 기울기를 한번만 계산하여 모델의 parameter 를 업데이트 하는 방법입니다.

​			장점		

​				전체 데이터에 대해 업데이트가 한번에 이루어지기 때문에 후술할 SGD 보다 업데이트 횟수가 적습니다. 

​				전체 데이터에 대해 error gradient 를 계산하기 때문에 수렴이 안정적으로 됩니다.

​				병렬 처리에 유리합니다.

​			단점

​				한 스텝에 모든 학습 데이터 셋을 사용하므로 학습이 오래 걸립니다.

​				전체 학습 데이터에 대한 error 를 모델의 업데이트가 이루어지기 전까지 축적해야 하므로 더 많은 메모리가 필요합니다.

​				local optimal 상태가 되면 빠져나오기 힘듦

			#### 		Stochastic gradient descent

​			추출된 데이터 한 개에 대해서 error gradient 를 계산하고, Gradient descent 알고리즘을 적용하는 방법입니다.

​			![image-20210909225341879](Image\SGD_1.png)

​			장점

​				위 그림에서 보이듯이 Shooting 이 일이나기 때문에 local optimal 에 빠질 리스크가 적다.

​				step 에 걸리는 시간이 짧아서 수렴속도가 빠릅니다.

​			단점

​				Global optimal 을 찾지 못 할 수도 있습니다.

​				데이터를 한개씩 처리하기에 GPU의 성능을 전부 활용할 수 없습니다.

#### 		Mini-batch gradient descent

​			![image-20210909230138709](Image\MSGD_1.png)

​			전체 데이터셋에서 뽑은 Mini-batch 안의 데이터 m 개에 대해서 각 데이터에 대한 기울기를 m 개 구한 뒤, 그것의 평균 기울기를

​			통해서 모델을 업데이트 하는 방법입니다.



​			전체 데이터 셋을 여러개의 mini-batch로 나누어 한 개의 mini-batch 마다 기울기를 구하고 모델을 업데이트 하는 것 입니다.

​			예를 들어 전체 데이터가 1000개 인데 batch size 를 10으로 하면 100개의 mini-batch가 생성이 되는 것으로 100 iteration 동안

​			모델이 업데이트 되며 1 epoch 가 끝납니다.



​			장점

​				BGD 보다 local optimal 에 빠질 리스크가 적습니다.

​				SGD 보다 병렬처리에 유리합니다.

​				전체 학습데이터가 아닌 일부분의 학습데이터만 사용하기 때문에 메모리 사용히 BGD 보다 적습니다.

​			단점

​				에러에 대한 정보를 mini-batch 크기 만큼 축적해서 계산하기 때문에 SGD 보다 메모리 사용이 높습니다.

​			batch size 는 2의 거듭제곱으로 해주는 것이 좋습니다. GPU의 메모리가 2의 거듭제곱 이라 batch size 를 2의 거듭제곱으로

​			해주는 것이 효율에 좋습니다.

###		Momentum

​		Momentum 은 Gradient descent 기반의 optimization algorithm 입니다. 

​																								![image-20210909231223312](Image\Momentum_1.png)

​		L : loss function value

​		W : weights

​		η : learning rate

​		α : 가속도 같은 역할을 하는 hyper parameter

​																						![image-20210909231404523](Image\Momentum_2.png)

​		첫 번째 스텝의 기울기가 5, 두 번째 스텝의 기울기가 3인 경우 학습률이 0.1 일때 가중치는 -0.3 만큼 변화합니다.

​		Momentum을 적용하면

​																		![](Image\Momentum_3.png)																																	

​		일반적인 GD 보다 두번째 스텝에서 -0.45 만큼 가중치가 더 변화하는 것을 알 수 있습니다.

​																![		](Image\Momentum_4.png)		

​		GD 를 사용했다면 갔을 거리보다 Momentum 을 사용했을 때 이동하려던 방향으로 스텝을 더 멀리 뻗습니다.

​	![image-20210909231714807](Image\Momentum_5.png)

​		Momentum 은 아래 그림처럼 실제 공이 굴러가듯이 가중치가 이동하게 됩니다.

#### 		Nesterov Accelerated Gradient

#### 					AdaGrad

#### 		Adadelta

	#### 		RMSprop

#### 		Adam
