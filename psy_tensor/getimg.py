#!/usr/bin/env python
import os,sys,string
import datetime,time

now = datetime.datetime(year=2016,month=7,day=7,hour=23,minute=00,second=0,microsecond=0)
dt = datetime.timedelta(minutes=10)
for d in range(0, 3000):
  #cmd = "wget \"http://www.kma.go.kr/cgi-bin/rdr/nph-rdr_cmp_cappi_img2?RDR&10&R&X&R&0000000_0000000&R&&C&640&XXA&%s&0&m&30&1&I\" -O %s.png" % (d, d)
  #os.system(cmd)
  s = now.strftime("%Y%m%d%H%M")
  pat = "%s.png" % (s)
  print pat
  if os.path.exists(pat):
    now -= dt
    continue

  cmd = "wget \"http://www.kma.go.kr/cgi-bin/rdr/nph-rdr_cmp_cappi_img2?RDR&10&R&X&R&0000000_0000000&R&&C&640&XXA&%s&0&m&30&1&I\" -O %s.png" % (s, s)
  os.system(cmd)
  now -= dt
