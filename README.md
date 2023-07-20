# lane_detection
# 车道线检测
# 项目描述
给一张道路照片，检测并拟合出白色车道线。
# 检测结果展示
<img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img1.jpg" height="340px"><img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img2.jpg" width="350px"><img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img3.jpg" height="340px">  
<img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img4.jpg" width="366px"><img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img5.jpg" width="366px">
<img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img6.jpg" width="366px"><img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img7.jpg" width="366px">
<img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img8.jpg" width="366px"><img src="https://github.com/xingchizhang/lane_detection/blob/main/imgs/img9.jpg" width="366px">
# 算法原理
程序的整体流程总体分为两步，第一步是从图片中检测并提取直线，第二步对直线进行处理获取到“车道”信息。  
详细步骤如下：  
①图像->灰度图->高斯模糊->Canny边缘检测->Hough变换->直线  
②直线->清理颜色差异->清理杂线->清理外线->清理短线->合并直线->清理短线->合并直线->车道  
