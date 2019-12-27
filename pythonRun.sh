run() {
    echo "Running... $*"
    python3 ssdDetector_expr.py $* -n gl552vw
}
if [[ "${1}" == "t" ]]; then
    echo "Test all conditions"
    run -f 300 -s
    run -f 300 -s -g
    run -f 300
    run -f 300 -g
else
    echo "Run only"
    run -f 10 -s -g
fi
