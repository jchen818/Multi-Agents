import pytest, sys
sys.path.insert(0, "tools/labcli")
from server import check_command

@pytest.mark.parametrize("cmd", [
    "amd-smi metric", "rocm-smi --showtemp", "dmesg -T --level=err,warn",
    "journalctl -k -n 200 --no-pager", "lspci -nn", "cat /proc/meminfo", "modinfo amdgpu", "cat /opt/rocm/.info/version", "ip -br addr", "ethtool -S eth0"])
def test_allowed(cmd):
    assert check_command(cmd)

@pytest.mark.parametrize("cmd", [
    "rm -rf /", "reboot", "dmesg -C", "dmesg -Tc", "journalctl -f", "journalctl --vacuum-size=1M",
    "rocm-smi --setfan 100", "rocm-smi -r", "amd-smi reset -G", "amd-smi set --power-cap 500",
    "cat /etc/shadow", "cat /proc/../etc/shadow", "cat /opt/rocm/../../etc/shadow", "modinfo amdgpu | grep version", "ip link set eth0 down", "ethtool -s eth0 speed 100",
    "uptime; reboot", "lspci | sh", "cat $(echo /etc/shadow)", "dmesg > /tmp/x"])
def test_refused(cmd):
    with pytest.raises(ValueError):
        check_command(cmd)

