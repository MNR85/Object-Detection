compile() {
    echo "Compiling..."
    g++ -std=c++11 ssdDetectorM13_multiStage.cpp MNR_Net.cpp -lboost_system -lcaffe -lglog -lgflags -lpthread -lcudart $(pkg-config opencv --cflags --libs) -o ssd
}
run() {
    echo "Running... $1 $2 $3"
    ./ssd MobileNetSSD_deploy.prototxt MobileNetSSD_deploy.caffemodel ../part02.mp4 $1 $2 $3
}
if [[ "${1}" == "a" ]]; then
    echo "Compile and run"
    compile && run 1 p g
elif [[ "${1}" == "c" ]]; then
    echo "Compile only"
    compile
elif [[ "${1}" == "t" ]]; then
    echo "Test all conditions"
    run 300 s c
    run 300 s g
    run 300 p c
    run 300 p g
else
    echo "Run only"
    run 10 s c
fi
