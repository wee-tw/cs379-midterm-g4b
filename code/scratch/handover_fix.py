"""Fixes to the PYNQ handover guide after the review (10/30 20:50)."""
import sys

p = sys.argv[1]
t = open(p, encoding="utf-8").read()


def rep(old, new):
    global t
    assert old in t, old[:90]
    t = t.replace(old, new)


# prep: check python first
rep("""有 <b>Python 3.8 以上</b>（不用裝額外套件）和 <b>OpenSSH</b>（Windows 10/11 內建）。</span>""",
    """有 <b>Python 3.8 以上</b>（不用裝額外套件）和 <b>OpenSSH</b>（Windows 10/11 內建）。在終端機輸入 <code>python --version</code>（或 <code>py --version</code>）和 <code>ssh -V</code> 確認兩個都有。</span>""")

# general note for macOS / Linux at the top of the steps
rep("""<section>
  <h2>步驟</h2>
""", """<section>
  <h2>步驟</h2>
  <div class="note warn"><b class="tag">先看這裡</b><span>下面的指令以 Windows PowerShell 為例。macOS / Linux 的話，把指令裡的 <code>"$env:USERPROFILE\\.ssh\\id_ed25519_pynq"</code> 換成 <code>~/.ssh/id_ed25519_pynq</code>，其他照打。板子如果不是 <code>192.168.2.99</code>（接路由器的情況），每一行的 IP 都要換掉。</span></div>
""")

# step 2: Windows 10 prefix length, macOS
rep("""IP <code>192.168.2.1</code>，子網路遮罩 <code>255.255.255.0</code>，其他空白。板子預設是 <code>192.168.2.99</code>。</p>""",
    """IP <code>192.168.2.1</code>，子網路遮罩 <code>255.255.255.0</code>（Windows 10 問「子網路首碼長度」時填 <code>24</code>），其他空白。macOS：系統設定 → 網路 → 乙太網路 → 詳細資訊 → TCP/IP → 手動。板子預設是 <code>192.168.2.99</code>。</p>""")

# step 3: overwrite prompt
rep("""<p>先產生一把專用金鑰。問 passphrase 時<b>直接按兩次 Enter</b>（不要設密碼，程式才能自動登入）。</p>""",
    """<p>先產生一把專用金鑰。問 passphrase 時<b>直接按兩次 Enter</b>（不要設密碼，程式才能自動登入）；問 <code>Overwrite (y/n)?</code> 時輸入 <code>y</code>。</p>""")
rep("""<div class="note ok"><b class="tag">正常</b><span>印出 <code>key-ok</code>。macOS / Linux 把 <code>"$env:USERPROFILE\\.ssh\\id_ed25519_pynq"</code> 換成 <code>~/.ssh/id_ed25519_pynq</code>。</span></div>""",
    """<div class="note ok"><b class="tag">正常</b><span>印出 <code>key-ok</code>，而且沒有問密碼。</span></div>""")

# step 5: note the logging start time and the time format
rep("""      <li>把功率計的<b>型號、軟體名稱、取樣率</b>寫進紀錄表。格式我們會處理。</li>""",
    """      <li>把功率計的<b>型號、軟體名稱、取樣率</b>寫進紀錄表，再寫下<b>開始記錄時電腦時鐘的時間（到秒）</b>，以及記錄檔的時間欄是「日期時間」還是「從 0 開始的秒數」。其他格式我們會處理。</li>""")

# step 6: terminal on Windows 11, and precise throttle criteria
rep("""<p>在解壓縮出來的 <code>board</code> 資料夾裡開終端機（PowerShell：在資料夾空白處按住 Shift 再按右鍵 →「在這裡開啟 PowerShell 視窗」）。這一步會把程式、bitstream 和測試圖片複製到板子，跑 CNV-W1A1 的準確率，並測試「停時脈」功能。</p>""",
    """<p>在解壓縮出來的 <code>board</code> 資料夾裡開終端機（Windows 11：在資料夾空白處按右鍵 →「在終端機中開啟」；Windows 10：按住 Shift 再按右鍵 →「在這裡開啟 PowerShell 視窗」）。這一步會把程式、bitstream 和測試圖片複製到板子，跑 CNV-W1A1 的準確率，並測試「停時脈」功能。</p>""")
rep("""    <div class="note ok"><b class="tag">正常</b><span><code>accuracy</code> 那行的 <code>top1</code> 大約在 75–90%；每個窗都是 <code>ok=True</code>；功率計看到的功率大約 1–4 W，時脈越高越大。</span></div>
    <div class="note stop"><b class="tag">如果卡住</b><span>在 <code>probe_throttle</code> 或 <code>verify_throttle</code> 超過 2 分鐘沒有新的輸出，而且 <code>ping</code> 不通，代表停時脈之後重啟不了。這不會弄壞板子：拔掉電源再插回去，等 1 分鐘開機，之後所有指令都加上 <code>--no-throttle</code>，並在紀錄表寫下來。</span></div>""",
    """    <div class="note ok"><b class="tag">正常</b><span><code>accuracy</code> 那行的 <code>top1</code> 大約在 75–90%；<code>probe_throttle</code> 那行有 <code>"ok": true</code>，<code>verify_throttle</code> 那行有 <code>"halt_verified": true</code>；每個窗都是 <code>ok=True</code>；功率計看到的功率大約 1–4 W，時脈越高越大。</span></div>
    <div class="note warn"><b class="tag">停時脈不能用（沒有危險）</b><span><code>probe_throttle</code> 那行是 <code>"ok": false</code>，note 寫 <code>use the slowest-divider policy</code>；或 <code>verify_throttle</code> 那行是 <code>"halt_verified": false</code>。程式會繼續跑，其他量測都有效。步驟 7 的指令最後加上 <code>--no-throttle</code>，並在紀錄表寫下來。</span></div>
    <div class="note stop"><b class="tag">如果卡住</b><span>出現下面任何一種：<code>probe_throttle</code> 那行有 <code>"stuck": true</code>；之後的窗都是 <code>ok=False</code>，訊息寫 <code>PL access blocked</code>；超過 2 分鐘沒有新的輸出；出現 <code>agent exited</code>。處理方式：先在終端機按 <code>Ctrl+C</code> 停掉程式，拔掉板子電源再插回去，等 1 分鐘開機，然後<b>重跑這一步的指令並加上 <code>--no-throttle</code></b>，步驟 7 也要加。這不會弄壞板子，請在紀錄表寫下時間。</span></div>""")

# step 7: hang handling and no-throttle duration
rep("""    <ul>
      <li>步驟 6 卡住過，或印出 throttle 不可用時，指令最後加上 <code>--no-throttle</code>。</li>""",
    """    <ul>
      <li>步驟 6 卡住過，或停時脈不能用時，指令最後加上 <code>--no-throttle</code>（完整版會變成約 90 分）。</li>
      <li>跑到一半卡住（判斷方式同步驟 6）：<code>Ctrl+C</code>、拔電源重開、等 1 分鐘，加上 <code>--no-throttle</code> 重跑。已經量到的資料會留在原本的資料夾，不會不見。</li>""")

# step 9: delete root-owned files before removing sudo
rep("""<pre>ssh -i "$env:USERPROFILE\\.ssh\\id_ed25519_pynq" xilinx@192.168.2.99 "sudo rm /etc/sudoers.d/90-a-prime; rm -rf ~/a_prime"</pre>""",
    """<pre>ssh -i "$env:USERPROFILE\\.ssh\\id_ed25519_pynq" xilinx@192.168.2.99 "sudo rm -rf /home/xilinx/a_prime; sudo rm /etc/sudoers.d/90-a-prime"</pre>""")

# troubleshooting rows
rep("""      <tr><td>超過 2 分鐘沒有新輸出、<code>ping</code> 不通</td><td>拔電源再插回去，等 1 分鐘，加 <code>--no-throttle</code> 重跑。寫進紀錄表。</td></tr>""",
    """      <tr><td>超過 2 分鐘沒有新輸出、<code>"stuck": true</code>、<code>PL access blocked</code></td><td>按 <code>Ctrl+C</code>，拔電源再插回去，等 1 分鐘，加 <code>--no-throttle</code> 重跑。寫進紀錄表。</td></tr>""")

open(p, "w", encoding="utf-8").write(t)
print("guide patched")
