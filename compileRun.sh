if [[ "${1}" == "a" ]]
then
echo "Compile and run"
g++ -std=c++11 ssdDetectorM12_expr.cpp MNR_Net.cpp  -lboost_system -lcaffe -lglog -lgflags -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd && ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel ../part02.mp4
elif [[ "${1}" == "c" ]]
then
echo "Compile only"
g++ -std=c++11 ssdDetectorM12_expr.cpp MNR_Net.cpp  -lboost_system -lcaffe -lglog -lgflags -lpthread -lcudart `pkg-config opencv --cflags --libs` -o ssd
else
echo "Run only"
./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel ../part02.mp4
fi