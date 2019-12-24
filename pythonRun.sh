run() {
    echo "Running... $*"
    python3 ssdDetector_expr.py $*
}
if [[ "${1}" == "t" ]]; then
    echo "Test all conditions"
    run -f 10 -s
    run -f 10 -s -g
    run -f 10
    run -f 10 -g
else
    echo "Run only"
    run -f 10 -s -g
fi
