import json
import time
import requests
from lxml import etree

text = "this is a test sentence"
url = 'https://write-free.www.deepl.com/jsonrpc?method=LMT_handle_jobs'

headers = {
            "authority":"write-free.www.deepl.com",
            "method":"POST",
            "path":"/jsonrpc?method=LMT_split_text",
            "scheme":"https",
            "accept": "*/*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "content-length":"244",
            "content-type":"application/json",
            "cookie":"dapUid=3b11f6bd-773f-45c4-a636-2ae1b37f7a3a; LMTBID=v2|705501f4-1499-46b8-89e5-6d2185e2eb28|c8c0330bb0b00443573c724d4f5257af; privacySettings=%7B%22v%22%3A%221%22%2C%22t%22%3A1713830400%2C%22m%22%3A%22LAX_AUTO%22%2C%22consent%22%3A%5B%22NECESSARY%22%2C%22PERFORMANCE%22%2C%22COMFORT%22%2C%22MARKETING%22%5D%7D; userCountry=HK; INGRESSCOOKIE=921e09479c2f7382aab0c72ed8833516|a6d4ac311669391fc997a6a267dc91c0; releaseGroups=3613.WDW-267.2.2_3939.B2B-596.1.1_4854.DM-1255.2.5_8635.DM-1158.2.3_9580.DV-150.1.2_4650.AP-312.2.8_8041.DM-1581.2.2_8564.SEO-656.2.2_10272.AAEXP-8660.1.1_10290.AAEXP-8678.1.1_975.DM-609.2.3_10278.AAEXP-8666.2.1_10284.AAEXP-8672.1.1_10289.AAEXP-8677.1.1_1997.DM-941.2.3_2055.DM-814.2.3_3961.B2B-663.2.3_4298.ACL-538.2.2_5376.WDW-360.2.2_9026.DF-3894.2.3_9129.DM-1419.2.1_10285.AAEXP-8673.1.1_4321.B2B-679.2.2_6359.DM-1411.2.10_10269.AAEXP-8657.1.1_10281.AAEXP-8669.1.1_10298.AAEXP-8686.1.1_1585.DM-900.2.3_2464.DM-1175.2.2_3127.DM-1032.2.2_5707.TACO-104.2.2_6402.DWFA-716.2.3_7758.B2B-949.2.3_10292.AAEXP-8680.1.1_10379.DF-3874.1.1_1583.DM-807.2.5_4478.SI-606.2.3_6732.DF-3818.2.4_10274.AAEXP-8662.1.1_10280.AAEXP-8668.2.1_10381.DF-3974.2.2_2373.DM-1113.2.4_2962.DF-3552.2.6_4121.WDW-356.2.5_8288.DAL-972.2.1_8391.DM-1630.2.2_8776.DM-1442.2.2_2274.DM-952.2.2_6781.ACL-720.2.1_10276.AAEXP-8664.2.1_10295.AAEXP-8683.1.1_10297.AAEXP-8685.1.1_1483.DM-821.2.2_2413.DWFA-524.2.4_7584.TACO-60.2.2_10238.MTD-392.2.1_10271.AAEXP-8659.1.1_10277.AAEXP-8665.1.1_220.DF-1925.1.9_1577.DM-594.2.3_4322.DWFA-689.2.2_7616.DWFA-777.2.2_7617.DWFA-774.2.2_9546.TC-1165.2.4_976.DM-667.2.3_2656.DM-1177.2.2_3283.DWFA-661.2.2_4853.DF-3503.2.1_5560.DWFA-638.2.2_5562.DWFA-732.2.2_6727.B2B-777.2.2_8783.DF-3926.2.1_9579.DV-149.1.2_10273.AAEXP-8661.1.1_10288.AAEXP-8676.1.1_2345.DM-1001.2.2_7759.DWFA-814.2.2_9128.DM-1297.1.1_9824.AP-523.2.2_10279.AAEXP-8667.1.1_10286.AAEXP-8674.1.1_2455.DPAY-2828.2.2_8253.DWFA-625.2.2_10275.AAEXP-8663.2.1_10282.AAEXP-8670.1.1_10283.AAEXP-8671.1.1_10296.AAEXP-8684.1.1_10380.DF-3973.2.1_10382.DF-3962.2.1_1571.DM-791.2.4_1780.DM-872.2.2_7794.B2B-950.2.4_8392.DWFA-813.2.2_8393.DPAY-3431.2.2_8779.DF-3881.2.2_10291.AAEXP-8679.1.1_10294.AAEXP-8682.1.1_5719.DWFA-761.2.2_9548.DV-148.1.2_9855.WTT-1235.1.3_10270.AAEXP-8658.1.1_10287.AAEXP-8675.1.1_10293.AAEXP-8681.1.1; __cf_bm=NwmemrxEabBfa.4W9taDYvQmblE86Ln1tk8amvrEzD8-1716986913-1.0.1.1-JDehTkLV81e79q3FyhRePz0RozvSs0b2AG1I8QcVpf6Vxne4m2ybid2HGfhTW9UI2EcqCJo2MTui8hq.RxBJnQ; dapVn=10; dapSid=%7B%22sid%22%3A%2250d4468f-cfdb-4199-bdb8-83dd7ad10dd9%22%2C%22lastUpdate%22%3A1716986924%7D",
            "origin":"https://www.deepl.com",
            "priority":"u=1, i",
            "referer":"https://www.deepl.com/",
            "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        }

params = {
  "jobs": [
    {
      "kind": "default",
      "sentences": [
        {
          "text": "this is",
          "id": 1,
          "prefix": ""
        }
      ],
      "raw_en_context_before": [],
      "raw_en_context_after": [],
      "preferred_num_beams": 4,
      "write_variant_requests": [
        "main",
        "variants"
      ],
      "quality": "fast"
    }
  ],
  "lang": {
    "target_lang": "EN",
    "preference": {
      "weight": {},
      "default": "default"
    },
    "source_lang_computed": "EN"
  },
  "commonJobParams": {
    "regionalVariant": "en-GB",
    "mode": "write",
    "browserType": 1,
    "textType": "plaintext"
  },
  "timestamp": int(round(time.time() * 1000)) 
}
print(int(round(time.time() * 1000)) )
res = requests.post(url, data=json.dumps(params), headers=headers)
str_json = res.context.decode('utf-8')

