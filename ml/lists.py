"""
Reference lists used by the URL and email checks.

I've kept these in one place so they can be reviewed (and extended) without
digging through the feature code. They are intentionally small and offline:
the prototype must never call out to a reputation service or DNS, so all the
"knowledge" it has about domains lives here.

None of these lists is meant to be exhaustive. They cover common patterns
from published phishing guidance and from the kinds of URLs visible in the
PhiUSIIL phishing samples.
"""

# Two-part public suffixes. Without this, "bbc.co.uk" would be treated as the
# domain "co.uk" with subdomain "bbc", which breaks every domain-level feature
# for UK (and many other) sites. A full Public Suffix List would be more
# accurate, but tldextract needs a network download on first use and I wanted
# the app to work offline.
MULTI_PART_SUFFIXES = {
    "co.uk", "org.uk", "ac.uk", "gov.uk", "nhs.uk", "sch.uk", "ltd.uk", "plc.uk", "me.uk", "net.uk",
    "police.uk", "com.au", "net.au", "org.au", "edu.au", "gov.au", "co.nz", "org.nz", "ac.nz",
    "govt.nz", "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp", "co.kr", "or.kr", "ac.kr", "go.kr",
    "com.br", "net.br", "org.br", "gov.br", "com.cn", "net.cn", "org.cn", "edu.cn", "gov.cn",
    "co.in", "net.in", "org.in", "ac.in", "gov.in", "com.mx", "com.ar", "com.tr", "com.sg",
    "edu.sg", "com.my", "com.hk", "com.tw", "co.za", "org.za", "com.pe", "com.co", "com.ua",
    "com.pl", "com.ph", "com.vn", "co.id", "or.id", "ac.id", "go.id", "com.eg", "com.sa",
    "com.pk", "com.ng", "co.ke", "co.il", "co.th", "ac.th", "com.bd", "edu.pk", "gob.mx",
}

URL_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "bitly.com",
    "cutt.ly", "rebrand.ly", "shorturl.at", "rb.gy", "tiny.cc", "lnkd.in", "s.id", "t.ly",
    "v.gd", "qrco.de", "bl.ink", "short.io", "soo.gd", "clck.ru", "u.to", "x.co", "trib.al",
}

# Platforms where anyone can publish a page under the platform's domain.
# Plenty of genuine sites live on these, so this is a warning, not a verdict,
# but several of the PhiUSIIL phishing samples I looked at were hosted on
# them (web.app, firebaseapp.com, ipfs.io and similar).
FREE_HOSTING_DOMAINS = {
    "web.app", "firebaseapp.com", "duckdns.org", "000webhostapp.com", "weebly.com", "wixsite.com",
    "webflow.io", "github.io", "glitch.me", "netlify.app", "vercel.app", "herokuapp.com",
    "repl.co", "ngrok.io", "ngrok-free.app", "blogspot.com", "sites.google.com", "pages.dev",
    "workers.dev", "azurewebsites.net", "square.site", "godaddysites.com", "myshopify.com",
    "ipfs.io", "arweave.net", "r2.dev", "surge.sh", "framer.app", "gitbook.io", "no-ip.org",
    "ddns.net", "hopto.org", "zapto.org", "sytes.net", "serveo.net", "carrd.co", "jimdosite.com",
}

# Cheap or free TLDs of the kind that regularly appear on published
# "most abused TLD" lists. Again, a signal only.
SUSPICIOUS_TLDS = {
    "tk", "ml", "ga", "cf", "gq", "xyz", "top", "zip", "mov", "click", "link", "work", "rest",
    "country", "kim", "loan", "men", "date", "racing", "review", "stream", "download", "gdn",
    "win", "bid", "trade", "party", "science", "cam", "icu", "buzz", "monster", "cyou", "sbs",
    "cfd", "quest", "support", "live", "online", "site", "website", "shop", "club",
}

# Words that phishing URLs lean on to look official or create urgency.
SUSPICIOUS_KEYWORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "account", "update",
    "secure", "security", "banking", "confirm", "password", "webscr", "ebayisapi", "wallet",
    "unlock", "suspend", "billing", "invoice", "payment", "recover", "authenticate", "auth",
    "validate", "support", "service", "alert", "limited", "reactivate", "credential", "session",
    "customer", "helpdesk", "refund", "claim", "bonus", "free", "gift", "prize", "airdrop",
]

# Brands that are impersonated often enough to be worth checking for.
# Generic words like "office" or "blockchain" were left out on purpose:
# they caused false matches on ordinary domains (officedepot.com, etc.).
COMMON_BRANDS = [
    "paypal", "apple", "icloud", "microsoft", "office365", "outlook", "google", "gmail",
    "amazon", "netflix", "facebook", "instagram", "whatsapp", "linkedin", "dropbox", "docusign",
    "adobe", "chase", "wellsfargo", "bankofamerica", "citibank", "hsbc", "barclays", "lloyds",
    "natwest", "santander", "halifax", "monzo", "revolut", "coinbase", "binance", "metamask",
    "dhl", "fedex", "ups", "usps", "royalmail", "hmrc", "ebay", "steam", "roblox",
    "yahoo", "aol", "att", "verizon", "americanexpress", "amex", "mastercard", "visa", "interac",
    "bnpparibas", "ing", "tesco", "walmart", "costco", "telegram", "discord", "twitter", "meta",
]

# Real domains owned by those brands where the domain name isn't simply the
# brand. Without this, outlook.office.com and login.microsoftonline.com were
# being reported as "brand outside the registered domain", which is exactly
# the kind of false alarm that teaches users to ignore warnings.
OFFICIAL_BRAND_DOMAINS = {
    "office.com", "office365.com", "microsoftonline.com", "live.com", "outlook.com", "msn.com",
    "sharepoint.com", "onedrive.com", "windows.net", "azure.com", "bing.com", "skype.com",
    "googleusercontent.com", "gstatic.com", "googleapis.com", "youtube.com", "gmail.com",
    "icloud.com", "apple.com", "me.com", "amazonaws.com", "amazon.com", "primevideo.com",
    "paypalobjects.com", "paypal.me", "fb.com", "facebook.com", "instagram.com", "whatsapp.com",
    "whatsapp.net", "linkedin.com", "licdn.com", "x.com", "twitter.com", "netflix.net",
    "nflxvideo.net", "chase.com", "wellsfargo.com", "bankofamerica.com", "hsbc.co.uk",
    "barclays.co.uk", "lloydsbank.com", "natwest.com", "royalmail.com", "gov.uk", "ebay.co.uk",
    "steampowered.com", "steamcommunity.com", "yahoo.com", "aol.com", "dhl.com", "fedex.com",
    "ups.com", "usps.com", "dropbox.com", "docusign.net", "docusign.com", "adobe.com",
    "coinbase.com", "binance.com", "metamask.io", "telegram.org", "discord.com", "discord.gg",
}

# Webmail providers. A message signed "PayPal Security" but sent from one of
# these is a classic impersonation pattern.
FREE_MAIL_PROVIDERS = {
    "gmail.com", "yahoo.com", "outlook.com", "hotmail.com", "aol.com", "icloud.com", "mail.com",
    "gmx.com", "protonmail.com", "proton.me", "yandex.com", "zoho.com", "live.com", "msn.com",
}
