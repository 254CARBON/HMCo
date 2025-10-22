# Cloudflare Free Tier - Available Features for 254carbon.com

**Account Plan**: Free  
**Last Updated**: October 20, 2025

---

## ✅ Features Included in Free Plan

### 1. Cloudflare Tunnel (Zero Trust)
- **Status**: ✅ Fully Operational
- **Bandwidth**: Unlimited
- **Connections**: Unlimited
- **Users**: Up to 50 seats for Cloudflare Access
- **What You Get**:
  - Secure tunnel from your Kubernetes cluster to Cloudflare edge
  - No public IP exposure required
  - Automatic reconnection and failover
  - HTTP/2 protocol support
  - Metrics endpoint for monitoring

**Current Usage**: 2 tunnel replicas, 8 active connections

### 2. Cloudflare Access (SSO)
- **Status**: ✅ Configured (14 applications)
- **Free Limit**: 50 users
- **What You Get**:
  - Single Sign-On for all applications
  - Email domain authentication (@254carbon.com)
  - Email OTP (one-time password) login
  - Application-level access policies
  - Audit logs (basic)
  - Session management

**Current Usage**: 14 applications with @254carbon.com email policies

### 3. DNS Management
- **Status**: ✅ Active (14 records)
- **What You Get**:
  - Fast, reliable DNS hosting
  - DNSSEC support
  - Unlimited DNS queries
  - CNAME flattening
  - API access for automation

**Current Usage**: 14 CNAME records pointing to tunnel

### 4. SSL/TLS
- **Status**: ✅ Active
- **What You Get**:
  - Universal SSL certificate (Cloudflare edge)
  - Flexible SSL modes (Off, Flexible, Full, Full Strict)
  - Modern TLS (TLS 1.2, TLS 1.3)
  - HTTPS rewrites
  - Always Use HTTPS

**Current Setting**: Flexible SSL (can upgrade to Full Strict with proper origin certs)

### 5. DDoS Protection
- **Status**: ✅ Active (automatic)
- **What You Get**:
  - Unmetered DDoS mitigation
  - Protection against volumetric attacks
  - Automatic activation on attack
  - No bandwidth limits during attacks

**Coverage**: All *.254carbon.com domains protected

### 6. Performance
- **What You Get**:
  - Global CDN (275+ cities)
  - HTTP/2 and HTTP/3 support
  - Auto minification (HTML, CSS, JS)
  - Brotli compression
  - Mobile optimization

### 7. Analytics
- **What You Get**:
  - Basic web analytics (last 24 hours)
  - Traffic overview
  - Top paths, countries, traffic types
  - Threat analytics
  - Access audit logs (basic)

---

## ❌ Features NOT Available on Free Plan

### Security (Requires Paid Plans)
- **WAF (Web Application Firewall)**: Requires Pro ($20/month) or higher
  - OWASP ruleset protection
  - Custom rules
  - Managed rulesets
  
- **Rate Limiting**: Requires Pro plan or higher
  - Request-based throttling
  - Custom rate limit rules
  - Per-endpoint limits

- **Advanced DDoS**: Requires Business/Enterprise
  - Advanced L7 DDoS protection
  - Dedicated support during attacks

- **Bot Management**: Requires Enterprise
  - Bot score and detection
  - JavaScript challenge
  - CAPTCHA challenges

### Performance (Requires Paid Plans)
- **Argo Smart Routing**: Requires $5/month + usage
  - Intelligent routing for faster connections
  
- **Load Balancing**: Requires $5/month
  - Geographic steering
  - Health checks
  - Failover

### Other Paid Features
- **Extended Analytics**: Requires Pro or higher (7-30 days retention)
- **Page Rules**: Free plan has 3, Pro has 20+
- **Workers**: 100,000 requests/day free, then paid
- **Stream**: Video platform (paid only)

---

## 🛡️ Security Best Practices (Free Tier)

### What You Should Configure

#### 1. SSL/TLS Mode (Recommended: Full Strict)
```
Current: Flexible (Cloudflare ↔ Origin can be HTTP)
Recommended: Full (Strict) - Requires valid origin certificate

Steps:
1. Go to Cloudflare Dashboard → SSL/TLS → Overview
2. Change mode to "Full (Strict)"
3. Generate Cloudflare Origin Certificate
4. Install in Kubernetes as TLS secret
```

#### 2. Security Level
```
Current: Medium (default)
Recommended: High (for sensitive applications)

Steps:
1. Go to Security → Settings
2. Set Security Level to "High"
3. This increases challenge frequency for suspicious visitors
```

#### 3. Challenge Passage
```
Current: 30 minutes (default)
Recommended: 15 minutes (for higher security)

Steps:
1. Go to Security → Settings
2. Set Challenge Passage to 15 minutes
3. Visitors must re-verify more frequently
```

#### 4. Browser Integrity Check
```
Recommended: Enabled

Steps:
1. Go to Security → Settings
2. Enable "Browser Integrity Check"
3. Blocks common threats and modified browsers
```

#### 5. Email Obfuscation
```
Recommended: Enabled

Steps:
1. Go to Scrape Shield → Email Obfuscation
2. Enable to protect email addresses from harvesters
```

#### 6. IP Access Rules (Firewall)
```
Use Case: Block/allow specific countries or IPs

Steps:
1. Go to Security → WAF (then "Tools" tab)
2. IP Access Rules
3. Add allow/block rules for IPs or countries
4. Example: Block all except specific countries
```

---

## 💰 Cost Optimization

### Free Tier Limits
- **Bandwidth**: Unlimited
- **Requests**: Unlimited
- **Users (Access)**: 50 seats
- **Storage**: N/A (tunnel only)
- **Data Transfer**: Unlimited

### Staying Within Free Tier
- ✅ You're using: Tunnel, Access, DNS, SSL, basic security
- ✅ Current usage: Well within limits
- ✅ No overage charges on Free plan

### When to Consider Upgrading

**Pro Plan ($20/month)** - Consider if you need:
- WAF protection
- Advanced analytics (30 days)
- More page rules
- Mobile optimization
- Image optimization

**Business Plan ($200/month)** - Consider if you need:
- Advanced DDoS protection
- Custom SSL certificates
- 100% uptime SLA
- Prioritized support

**Enterprise** - For large-scale deployments requiring:
- Bot management
- China network
- Custom contracts
- Dedicated support team

---

## 🎯 What We're Using (Free Features)

### Active Features
1. ✅ **Cloudflare Tunnel**: 2 replicas, 8 connections
2. ✅ **Cloudflare Access**: 14 applications, @254carbon.com policy
3. ✅ **DNS Hosting**: 14 CNAME records
4. ✅ **Universal SSL**: Edge encryption
5. ✅ **DDoS Protection**: Automatic and unmetered
6. ✅ **CDN**: Global edge caching
7. ✅ **Analytics**: Basic traffic insights

### Recommended Configurations (No Cost)
1. ⚠️ Set SSL/TLS to "Full (Strict)" mode
2. ⚠️ Increase Security Level to "High"
3. ⚠️ Enable Browser Integrity Check
4. ⚠️ Set up IP Access Rules (if needed)
5. ⚠️ Configure Always Use HTTPS

---

## 📊 Feature Comparison

| Feature | Free | Pro | Business | Enterprise |
|---------|------|-----|----------|------------|
| Cloudflare Tunnel | ✅ Yes | ✅ Yes | ✅ Yes | ✅ Yes |
| Cloudflare Access | ✅ 50 users | ✅ Unlimited | ✅ Unlimited | ✅ Unlimited |
| DDoS Protection | ✅ Basic | ✅ Basic | ✅ Advanced | ✅ Advanced |
| WAF | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| Rate Limiting | ❌ No | ✅ Yes | ✅ Yes | ✅ Yes |
| SSL Certificates | ✅ Universal | ✅ + Advanced | ✅ + Custom | ✅ + Custom |
| Analytics Retention | 24 hours | 30 days | 30 days | As needed |
| Page Rules | 3 | 20 | 50 | 125 |
| Support | Community | Email | 24/7 | Dedicated |

**Our Status**: Free tier meets all current requirements ✅

---

## 🔗 Useful Free Tier Links

- **Cloudflare Plans**: https://www.cloudflare.com/plans/
- **Zero Trust Pricing**: https://www.cloudflare.com/products/zero-trust/
- **Free Features Guide**: https://developers.cloudflare.com/fundamentals/subscriptions-and-billing/
- **SSL/TLS Modes**: https://developers.cloudflare.com/ssl/origin-configuration/ssl-modes/

---

## 💡 Pro Tips for Free Plan

1. **Use Cloudflare Access Effectively**: You get 50 free users - use email domain policies to cover entire organization
2. **Leverage Universal SSL**: Free edge TLS is excellent, origin can use self-signed
3. **Use IP Access Rules**: Free firewall functionality - block unwanted traffic by country/IP
4. **Monitor Analytics Daily**: Only 24-hour retention, so check regularly
5. **Use Page Rules Wisely**: You get 3 free - use for critical redirects or caching
6. **Enable All Free Security**: Browser integrity check, email obfuscation, etc.

---

**Bottom Line**: The Cloudflare Free plan provides enterprise-grade tunnel, SSO, and security features at no cost. Perfect for your current setup! 🎉

