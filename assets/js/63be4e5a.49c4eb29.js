(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[520],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return d},kt:function(){return m}});var a=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function i(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);t&&(a=a.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,a)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?i(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):i(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function o(e,t){if(null==e)return{};var n,a,r=function(e,t){if(null==e)return{};var n,a,r={},i=Object.keys(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);for(a=0;a<i.length;a++)n=i[a],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var c=a.createContext({}),p=function(e){var t=a.useContext(c),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},d=function(e){var t=p(e.components);return a.createElement(c.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return a.createElement(a.Fragment,{},t)}},u=a.forwardRef((function(e,t){var n=e.components,r=e.mdxType,i=e.originalType,c=e.parentName,d=o(e,["components","mdxType","originalType","parentName"]),u=p(n),m=r,h=u["".concat(c,".").concat(m)]||u[m]||s[m]||i;return n?a.createElement(h,l(l({ref:t},d),{},{components:n})):a.createElement(h,l({ref:t},d))}));function m(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var i=n.length,l=new Array(i);l[0]=u;var o={};for(var c in t)hasOwnProperty.call(t,c)&&(o[c]=t[c]);o.originalType=e,o.mdxType="string"==typeof e?e:r,l[1]=o;for(var p=2;p<i;p++)l[p]=n[p];return a.createElement.apply(null,l)}return a.createElement.apply(null,n)}u.displayName="MDXCreateElement"},81242:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return o},contentTitle:function(){return c},metadata:function(){return p},toc:function(){return d},default:function(){return u}});var a=n(22122),r=n(19756),i=(n(67294),n(3905)),l=["components"],o={},c="WatchDog",p={unversionedId:"chips/600X/application/peripheral/samples/watchdog",id:"chips/600X/application/peripheral/samples/watchdog",isDocsHomePage:!1,title:"WatchDog",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/peripheral/samples/watchdog.md",sourceDirName:"chips/600X/application/peripheral/samples",slug:"/chips/600X/application/peripheral/samples/watchdog",permalink:"/chips/600X/application/peripheral/samples/watchdog",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"RTC",permalink:"/chips/600X/application/peripheral/samples/RTC"},next:{title:"Timer\u5b9a\u65f6\u5668",permalink:"/chips/600X/application/kernel/timer"}},d=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u51c6\u5907\u5de5\u4f5c",id:"\u51c6\u5907\u5de5\u4f5c",children:[]},{value:"\u521b\u5efa\u9879\u76ee",id:"\u521b\u5efa\u9879\u76ee",children:[]},{value:"\u4ee3\u7801\u5b9e\u73b0",id:"\u4ee3\u7801\u5b9e\u73b0",children:[{value:"\u7ec4\u4ef6\u914d\u7f6e",id:"\u7ec4\u4ef6\u914d\u7f6e",children:[]},{value:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790",id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790",children:[]}]},{value:"\u7f16\u8bd1\u548c\u70e7\u5f55",id:"\u7f16\u8bd1\u548c\u70e7\u5f55",children:[]}],s={toc:d};function u(e){var t=e.components,o=(0,r.Z)(e,l);return(0,i.kt)("wrapper",(0,a.Z)({},s,o,{components:t,mdxType:"MDXLayout"}),(0,i.kt)("h1",{id:"watchdog"},"WatchDog"),(0,i.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,i.kt)("p",null,"Watch Dog(\u770b\u95e8\u72d7)\u662f\u4e00\u79cd\u76d1\u63a7\u7cfb\u7edf\u8f6f\u4ef6\u7684\u8fd0\u884c\u72b6\u51b5\u7684\u624b\u6bb5\u3002\u7a33\u5b9a\u8fd0\u884c\u7684\u8f6f\u4ef6\u4f1a\u5728\u6267\u884c\u5b8c\u7279\u5b9a\u6307\u4ee4\u540e\u8fdb\u884c\u5582\u72d7\uff0c\u82e5\u5728\u4e00\u5b9a\u5468\u671f\u5185\u770b\u95e8\u72d7\u6ca1\u6709\u6536\u5230\u6765\u8f6f\u4ef6\u7a0b\u5e8f\u7684\u5582\u72d7\u4fe1\u53f7\uff0c\u5219\u8ba4\u4e3a\u8f6f\u4ef6\u8fd0\u884c\u51fa\u73b0\u4e86\u5f02\u5e38\uff0c\u8fdb\u5165\u4e2d\u65ad\u5904\u7406\u7a0b\u5e8f\u6216\u5f3a\u5236\u7cfb\u7edf\u590d\u4f4d\uff0c\u672c\u7ae0\u8282\u901a\u8fc7\u793a\u4f8b\u4ecb\u7ecd\u770b\u95e8\u72d7\u7684\u57fa\u672c\u4f7f\u7528\u65b9\u6cd5\u3002"),(0,i.kt)("h2",{id:"\u51c6\u5907\u5de5\u4f5c"},"\u51c6\u5907\u5de5\u4f5c"),(0,i.kt)("p",null,"\u672c\u793a\u4f8b\u57fa\u4e8e ",(0,i.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u6765\u5b9e\u73b0\uff0c\u9700\u8981\u505a\u5982\u4e0b\u51c6\u5907\uff1a"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u51c6\u5907\u4e00\u4e2a",(0,i.kt)("inlineCode",{parentName:"li"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f"),(0,i.kt)("li",{parentName:"ul"},"\u901a\u8fc7\u4e32\u53e3\u8fde\u63a5PC\u7aef\u67e5\u770b\u65e5\u5fd7")),(0,i.kt)("h2",{id:"\u521b\u5efa\u9879\u76ee"},"\u521b\u5efa\u9879\u76ee"),(0,i.kt)("p",null,"\u901a\u8fc7Lisa\u547d\u4ee4\u521b\u5efa\u9879\u76ee\uff1a"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,i.kt)("p",null,(0,i.kt)("img",{src:n(89072).Z}),"\n\u4f9d\u6b21\u6309\u4e00\u4e0b\u76ee\u5f55\u9009\u62e9\u5b8c\u6210watchdog sample\u521b\u5efa\uff1a  "),(0,i.kt)("blockquote",null,(0,i.kt)("p",{parentName:"blockquote"},"boards/ --\x3e csk6/ --\x3e driver/ --\x3e wdt")),(0,i.kt)("h2",{id:"\u4ee3\u7801\u5b9e\u73b0"},"\u4ee3\u7801\u5b9e\u73b0"),(0,i.kt)("h3",{id:"\u7ec4\u4ef6\u914d\u7f6e"},"\u7ec4\u4ef6\u914d\u7f6e"),(0,i.kt)("p",null,"\u5728prj.conf\u6587\u4ef6\u4e2d\u6253\u5f00uart\u529f\u80fd\u914d\u7f6e:"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"# watchdog \u914d\u7f6e\nCONFIG_WATCHDOG=y\nCONFIG_WDT_CSK6=y\n")),(0,i.kt)("h3",{id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790"},"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790"),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u5e94\u7528\u903b\u8f91\uff1a"),"  "),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},"\u542f\u52a8\u770b\u95e8\u72d7\uff0c\u8bbe\u7f6e\u8d85\u65f6\u65f6\u95f4\u4e3a16S\u3002"),(0,i.kt)("li",{parentName:"ul"},"10S\u540e\u8fdb\u884c\u5582\u72d7\u64cd\u4f5c\uff0c\u8d85\u65f6\u65f6\u95f4\u6e05\u96f6\u91cd\u65b0\u8ba1\u65f6\u3002"),(0,i.kt)("li",{parentName:"ul"},"\u8ba1\u65f616S\u8d85\u65f6\u540e\u89e6\u53d1\u4e2d\u65ad\uff0c\u7b2c\u4e00\u6b21\u89e6\u53d1\u4e2d\u65ad\u65f6\u505a\u4e00\u6b21",(0,i.kt)("inlineCode",{parentName:"li"},"feed watchdog"),"\u64cd\u4f5c\uff0c\u7b2c\u4e8c\u6b21\u89e6\u53d1\u4e2d\u65ad\u4e0d\u505a\u4efb\u4f55\u5904\u7406\uff0c\u7b49\u5f85",(0,i.kt)("inlineCode",{parentName:"li"},"watchdog"),"\u91cd\u542f\u7cfb\u7edf\u3002")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u4f7f\u7528\u7684API\u63a5\u53e3\uff1a"),(0,i.kt)("br",{parentName:"p"}),"\n","\u793a\u4f8b\u4e2d\u4e3b\u8981\u7528\u5230\u4ee5\u4e0bWatchDog API\u63a5\u53e3\uff0c\u66f4\u591aAPI\u63a5\u53e3\u53ef\u4ee5\u5728zephyr\u5b98\u7f51",(0,i.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__watchdog__interface.html"},"Watchdog Interface API"),"\u4e2d\u770b\u5230\u3002"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},"/*\u542f\u52a8\u770b\u95e8\u72d7*/\nint wdt_setup(const struct device * dev, uint8_t options)   \n\n/*feed wdt\u64cd\u4f5c*/\nint wdt_feed(const struct device * dev, int channel_id)\n\n/*\u8bbe\u7f6e\u8d85\u65f6\u65f6\u95f4\uff0c\u8d85\u65f6\u540e\u89e6\u53d1\u4e2d\u65ad*/\nint wdt_install_timeout(const struct device *dev,  const struct wdt_timeout_cfg *cfg)\n")),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0:"),"  "),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},'void main(void)\n{\n    /* \u901a\u8fc7\u8bbe\u5907\u6811\u83b7\u53d6watchdog\u8bbe\u5907\u5b9e\u4f8b */\n    const struct device *wdt;\n    wdt = DEVICE_DT_GET(DT_NODELABEL(wdt));\n\n    struct wdt_timeout_cfg wdt_config = {\n        .flags = WDT_FLAG_RESET_SOC,\n        .window.min = 0,\n        .window.max = WDT_MAX_WINDOW,\n    };\n\n    /* \u8bbe\u7f6ewatchdog\u56de\u8c03\u4e8b\u4ef6 */\n    wdt_config.callback = wdt_callback;\n\n    /* \u8bbe\u7f6ewatchdog\u8d85\u65f6\u65f6\u95f4\uff0c\u8d85\u65f6\u540e\u89e6\u53d1\u4e2d\u65ad */\n    wdt_channel_id = wdt_install_timeout(wdt, &wdt_config);\n\n    /* \u542f\u52a8watchdog*/\n    err = wdt_setup(wdt, WDT_OPT_PAUSE_HALTED_BY_DBG);\n\n    /* waiting for Feeding watchdog */\n    uint32_t count = 0;\n    for (int i = 0; i < WDT_FEED_TRIES; ++i) {\n        k_msleep(1000);\n        printk("Waiting for feed watchdog, %ds...\\n", ++count);\n    }\n    /* \u6267\u884c\u4e00\u6b21Feed watchdog \u64cd\u4f5c*/\n    wdt_feed(wdt, wdt_channel_id);\n\n    /* \u7b49\u5f85watchdog\u91cd\u542f\u7cfb\u7edf */\n    count = 0;\n    while (1) {\n        k_msleep(1000);\n        printk("Waiting for reset, %ds...\\n", ++count);\n    }\n}\n')),(0,i.kt)("p",null,(0,i.kt)("strong",{parentName:"p"},"\u4e2d\u65ad\u5904\u7406\uff1a"),"\n\u5728\u7b2c\u4e00\u6b21\u8d85\u65f6\u8fdb\u5165\u4e2d\u65ad\u65f6\u6267\u884c\u4e00\u6b21",(0,i.kt)("inlineCode",{parentName:"p"},"Feed watchdog"),"\u64cd\u4f5c\uff0c16S\u540e\u518d\u6b21\u89e6\u53d1\u4e2d\u65ad\uff0c\u6b64\u65f6\u4e0d\u505a\u4efb\u4f55\u5904\u7406\uff0c\u7b49\u5f85watchdog\u91cd\u542f\u7cfb\u7edf\u3002"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre",className:"language-c"},'static void wdt_callback(const struct device *wdt_dev, int channel_id)\n{   \n    static bool is_handled = false;\n\n    if (!is_handled) {\n        is_handled = !is_handled;\n        wdt_feed(wdt_dev, channel_id);\n        printk("Handled watchdog interrupt and feed the dog.System will reset after 16s.\\n");\n    } else {\n        printk("Handled watchdog interrupt again, wait for reset.\\n");\n    }\n}\n')),(0,i.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,i.kt)("div",{parentName:"div",className:"admonition-heading"},(0,i.kt)("h5",{parentName:"div"},(0,i.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,i.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,i.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,i.kt)("div",{parentName:"div",className:"admonition-content"},(0,i.kt)("p",{parentName:"div"},"\u8fdb\u5165\u4e2d\u65ad\u540e\u4e0d\u505a\u4efb\u4f55\u5904\u7406\u5230watchdog\u89e6\u53d1\u7cfb\u7edf\u91cd\u542f\u5927\u69820.5S\uff0c\u7531\u5e95\u5c42\u9a71\u52a8\u8bbe\u7f6e\u3002"))),(0,i.kt)("h2",{id:"\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"\u7f16\u8bd1"),"  ")),(0,i.kt)("p",null,"\u5728app\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4ee5\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"\u70e7\u5f55"),"     ")),(0,i.kt)("p",null,(0,i.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u901a\u8fc7USB\u8fde\u63a5PC\uff0c\u901a\u8fc7\u70e7\u5f55\u6307\u5b8c\u6210\u70e7\u5f55\uff1a"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"lisa zep flash --runner pyocd\n")),(0,i.kt)("ul",null,(0,i.kt)("li",{parentName:"ul"},(0,i.kt)("strong",{parentName:"li"},"\u67e5\u770b\u7ed3\u679c"))),(0,i.kt)("p",null,"\u53ef\u901a\u8fc7lisa\u63d0\u4f9b\u7684",(0,i.kt)("inlineCode",{parentName:"p"},"lisa term"),"\u547d\u4ee4\u67e5\u770b\u65e5\u5fd7\uff1a"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"lisa term\n")),(0,i.kt)("p",null,"\u6216\u8005\u5c06",(0,i.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u7684\u65e5\u5fd7\u4e32\u53e3",(0,i.kt)("inlineCode",{parentName:"p"},"A03 TX A02 RX"),"\u63a5\u4e32\u53e3\u677f\u8fde\u63a5\u7535\u8111\uff0c\u5728\u7535\u8111\u7aef\u4f7f\u7528\u4e32\u53e3\u8c03\u8bd5\u52a9\u624b\u67e5\u770b\u65e5\u5fd7\uff0c\u6ce2\u7279\u7387\u4e3a115200\u3002"),(0,i.kt)("pre",null,(0,i.kt)("code",{parentName:"pre"},"*** Booting Zephyr OS build fd83997719ed  ***\nWatchdog sample application, May 30 2022 22:14:35\nAttempting to test pre-reset callback, 0x18001381\nwdt_channel_id=16252.\nWaiting for feed watchdog, 1s...\n...\nWaiting for feed watchdog, 10s...\nWaiting for reset, 1s...\n\n...\nWaiting for reset, 16s...\nHandled watchdog interrupt and feed the dog.System will reset after 16s.\nWaiting for reset, 1s...\n...\nWaiting for reset, 16s...\nHandled watchdog interrupt again, wait for reset.\n\n*** Booting Zephyr OS build fd83997719ed  ***\nWatchdog sample application, May 30 2022 22:15:20\n...\n")),(0,i.kt)("p",null," \u4ece\u65e5\u5fd7\u53ef\u4ee5\u770b\u5230\uff0cwatchdog\u5728\u7b2c\u4e00\u6b21\u89e6\u53d1\u4e2d\u65ad\u65f6\u6267\u884c\u4e86\u4e00\u6b21",(0,i.kt)("inlineCode",{parentName:"p"},"feed watchdog"),"\u64cd\u4f5c\uff0c\u7b2c\u4e8c\u6b21\u89e6\u53d1\u4e2d\u65ad\u4e0d\u505a\u4efb\u4f55\u5904\u7406\uff0c",(0,i.kt)("inlineCode",{parentName:"p"},"watchdog"),"\u91cd\u542f\u4e86\u7cfb\u7edf\uff0c\u7b26\u5408\u793a\u4f8b\u7684\u5b9e\u73b0\u9884\u671f\u3002"))}u.isMDXComponent=!0},89072:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_create01-4190be564317cfb86204637eaa81c2cb.png"}}]);