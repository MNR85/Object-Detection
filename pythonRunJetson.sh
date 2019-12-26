run() {
    echo "Running... $*"
    python ssdDetector_expr.py $* -n tegraK1
}
if [[ "${1}" == "t" ]]; then
    echo "Test all conditions"
    run -f 500 -s
    run -f 500 -s -g
    run -f 500
    run -f 500 -g
else
    echo "Run only"
    run -f 10 -g
fi
