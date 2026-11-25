#!/usr/bin/env bash
# Step 0 - can this server reach the data hosts? (~10 s)   usage: bash 0_check_network.sh
set -uo pipefail
source "$(dirname "$0")/env.sh"
LATIS="${FISM_LATIS_URL:-https://lasp.colorado.edu/lisird/latis/dap/fism_flare_hr.csv}"
LEG="${FISM_GOES_LEGACY_BASE:-https://www.ncei.noaa.gov/data/goes-space-environment-monitor/access/science/xrs}"
GR="${FISM_GOESR_BASE:-https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/goes}"
F107="${FISM_F107_URL:-https://spaceweather.gc.ca/solar_flux_data/daily_flux_values/fluxtable.txt}"
# 2 minutes x 2 bins of FISM2 around the X2.2 flare of 2011-02-15 (query operators percent-encoded)
Q="time%3E=2011-02-15T01:55&time%3C2011-02-15T01:57&wavelength%3E9&wavelength%3C9.2"
declare -a NAMES=("LaTiS FISM2 (lasp.colorado.edu)" "GOES 8-15 XRS (ncei.noaa.gov)" "GOES-R XRS (data.ngdc.noaa.gov)" "F10.7 (spaceweather.gc.ca)" "PyPI (pip install)")
declare -a URLS=("$LATIS?$Q" "$LEG/goes15/xrsf-l2-avg1m_science/2011/02/" "$GR/goes16/l2/data/xrsf-l2-avg1m_science/2017/09/" "$F107" "https://pypi.org/simple/numpy/")
fail=0
out="$LOGDIR/0_network.txt"; : > "$out"
rm -f "$LOGDIR"/0_resp_*.txt
for i in "${!URLS[@]}"; do
  code=$(curl -s -o "$LOGDIR/0_resp_$i.txt" -w '%{http_code}' --max-time 40 "${URLS[$i]}" 2>/dev/null)
  code=${code:-000}
  if [ "$code" = "200" ]; then st="OK  "; else st="FAIL"; fail=1; fi
  printf '%s  HTTP %s  %s\n' "$st" "$code" "${NAMES[$i]}" | tee -a "$out"
done
echo "--- first lines of the LaTiS answer (format check) ---" | tee -a "$out"
head -4 "$LOGDIR/0_resp_0.txt" 2>/dev/null | tee -a "$out"
echo "--- GOES-15 listing: number of .nc files for 2011-02 ---" | tee -a "$out"
grep -o 'sci_xrsf-l2-avg1m_g15_d[0-9]*_v[0-9-]*\.nc' "$LOGDIR/0_resp_1.txt" 2>/dev/null | sort -u | wc -l | tee -a "$out"
if [ "$fail" = "1" ]; then
  echo "Some hosts are unreachable (proxy/firewall?). If the server needs a proxy:" | tee -a "$out"
  echo "  export HTTPS_PROXY=http://<proxy>:<port>   and rerun." | tee -a "$out"
  exit 1
fi
log "network OK"
