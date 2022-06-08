(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[7],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return u},kt:function(){return k}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function c(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function i(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var p=r.createContext({}),o=function(e){var t=r.useContext(p),n=t;return e&&(n="function"==typeof e?e(t):c(c({},t),e)),n},u=function(e){var t=o(e.components);return r.createElement(p.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,l=e.originalType,p=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),d=o(n),k=a,m=d["".concat(p,".").concat(k)]||d[k]||s[k]||l;return n?r.createElement(m,c(c({ref:t},u),{},{components:n})):r.createElement(m,c({ref:t},u))}));function k(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=n.length,c=new Array(l);c[0]=d;var i={};for(var p in t)hasOwnProperty.call(t,p)&&(i[p]=t[p]);i.originalType=e,i.mdxType="string"==typeof e?e:a,c[1]=i;for(var o=2;o<l;o++)c[o]=n[o];return r.createElement.apply(null,c)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},35505:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return i},contentTitle:function(){return p},metadata:function(){return o},toc:function(){return u},default:function(){return d}});var r=n(22122),a=n(19756),l=(n(67294),n(3905)),c=["components"],i={},p="RTC",o={unversionedId:"chips/600X/application/peripheral/samples/RTC",id:"chips/600X/application/peripheral/samples/RTC",isDocsHomePage:!1,title:"RTC",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/peripheral/samples/RTC.md",sourceDirName:"chips/600X/application/peripheral/samples",slug:"/chips/600X/application/peripheral/samples/RTC",permalink:"/chips/600X/application/peripheral/samples/RTC",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"ADC",permalink:"/chips/600X/application/peripheral/samples/ADC"},next:{title:"WatchDog",permalink:"/chips/600X/application/peripheral/samples/watchdog"}},u=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u51c6\u5907\u5de5\u4f5c",id:"\u51c6\u5907\u5de5\u4f5c",children:[]},{value:"\u521b\u5efa\u9879\u76ee",id:"\u521b\u5efa\u9879\u76ee",children:[]},{value:"\u4ee3\u7801\u5b9e\u73b0",id:"\u4ee3\u7801\u5b9e\u73b0",children:[{value:"\u7ec4\u4ef6\u914d\u7f6e",id:"\u7ec4\u4ef6\u914d\u7f6e",children:[]},{value:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790",id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790",children:[]}]},{value:"\u7f16\u8bd1\u548c\u70e7\u5f55",id:"\u7f16\u8bd1\u548c\u70e7\u5f55",children:[]}],s={toc:u};function d(e){var t=e.components,i=(0,a.Z)(e,c);return(0,l.kt)("wrapper",(0,r.Z)({},s,i,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"rtc"},"RTC"),(0,l.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,l.kt)("p",null,"RTC(Real Time Clock)\u5b9e\u65f6\u65f6\u949f\u662f\u5e94\u7528\u5f00\u53d1\u4e2d\u5e38\u7528\u7684\u5916\u8bbe\uff0c\u672c\u7ae0\u8282\u63d0\u4f9b\u4e00\u4e2a\u57fa\u4e8ecounter\u8ba1\u6570\u5668API\u63a5\u53e3\u5b9e\u73b0\u95f9\u949f\u529f\u80fd\u7684\u793a\u4f8b\u6765\u4ecb\u7ecdRTC\u7684\u4f7f\u7528\u3002"),(0,l.kt)("p",null,"CSK6 RTC\u9a71\u52a8\u5177\u6709\u5982\u4e0b\u7279\u6027\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u65f6\u949f\u6e90\uff08\u65f6\u949f\u5206\u9891\u5668\u524d\uff09\u9891\u7387\u4e3a32.768KHz\u3002"),(0,l.kt)("li",{parentName:"ul"},"\u53ef\u4ee5\u8bbe\u7f6e\u79d2\u3001\u5206\u548c\u5c0f\u65f6\u7684\u5b9a\u65f6\u4e2d\u65ad\u3002")),(0,l.kt)("h2",{id:"\u51c6\u5907\u5de5\u4f5c"},"\u51c6\u5907\u5de5\u4f5c"),(0,l.kt)("p",null,"\u672c\u793a\u4f8b\u57fa\u4e8e ",(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u6765\u5b9e\u73b0\uff0c\u9700\u8981\u505a\u5982\u4e0b\u51c6\u5907\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u51c6\u5907\u4e00\u4e2a",(0,l.kt)("inlineCode",{parentName:"li"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f"),(0,l.kt)("li",{parentName:"ul"},"\u901a\u8fc7\u4e32\u53e3\u8fde\u63a5PC\u7aef\u67e5\u770b\u65e5\u5fd7")),(0,l.kt)("h2",{id:"\u521b\u5efa\u9879\u76ee"},"\u521b\u5efa\u9879\u76ee"),(0,l.kt)("p",null,"\u901a\u8fc7Lisa\u547d\u4ee4\u521b\u5efa\u9879\u76ee\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,l.kt)("p",null,(0,l.kt)("img",{src:n(89072).Z}),"\n\u4f9d\u6b21\u6309\u4e00\u4e0b\u76ee\u5f55\u9009\u62e9\u5b8c\u6210adc sample\u521b\u5efa\uff1a  "),(0,l.kt)("blockquote",null,(0,l.kt)("p",{parentName:"blockquote"},"boards \u2192 csk6 \u2192 driver \u2192 rtc")),(0,l.kt)("h2",{id:"\u4ee3\u7801\u5b9e\u73b0"},"\u4ee3\u7801\u5b9e\u73b0"),(0,l.kt)("h3",{id:"\u7ec4\u4ef6\u914d\u7f6e"},"\u7ec4\u4ef6\u914d\u7f6e"),(0,l.kt)("p",null,"\u5728prj.conf\u6587\u4ef6\u4e2d\u6253\u5f00uart\u529f\u80fd\u914d\u7f6e:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-shell"},"CONFIG_STDOUT_CONSOLE=y\nCONFIG_PRINTK=y\n\n#\u8ba1\u65f6\u5668\nCONFIG_COUNTER=y\n\n#CSK6\u8ba1\u65f6\u5668\nCONFIG_COUNTER_CSK6=y\n")),(0,l.kt)("h3",{id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790"},"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u5e94\u7528\u5b9e\u73b0\u903b\u8f91\uff1a"),"\n\u57fa\u4e8eCSK6\u9a71\u52a8\u63d0\u4f9b\u7684RTC\u65f6\u949f\uff0c\u8c03\u7528counter\u8ba1\u6570\u5668\u63a5\u53e3\u8bbe\u7f6e\u4e00\u4e2a2S\u7684\u95f9\u949f\uff0c\u95f9\u949f\u65f6\u95f4\u5230\u65f6\u89e6\u53d1\u4e2d\u65ad\u56de\u8c03\uff0c\u5e76\u5728\u56de\u8c03\u4e2d\u505a\u54cd\u5e94\u7684\u5904\u7406\u3002\n",(0,l.kt)("strong",{parentName:"p"},"\u4f7f\u7528\u7684API\u63a5\u53e3\uff1a"),(0,l.kt)("br",{parentName:"p"}),"\n","\u793a\u4f8b\u4e2d\u4e3b\u8981\u7528\u5230\u4ee5\u4e0bcounter\u8ba1\u6570\u5668API\u63a5\u53e3\uff0c\u66f4\u591a\u8ba1\u6570\u5668API\u63a5\u53e3\u53ef\u4ee5\u5728zephyr\u5b98\u7f51",(0,l.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__counter__interface.html"},"counter interface APIs"),"\u4e2d\u770b\u5230\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-c"},"/*\u83b7\u53d6\u6700\u5927\u7684\u5b9a\u65f6\u65f6\u95f4\uff0c\u5728csk6\u4e0atop value\u4e3a:4296508s(1193h) */\nuint32_t counter_get_top_value(const struct device * dev)   \n\n/* \u542f\u52a8\u8ba1\u65f6\u5668 */\ncounter_start(rtc); \n\n/* \u83b7\u53d6\u5f53\u524d\u8ba1\u6570\u5668\u503c */\nint counter_get_value(const struct device * dev, uint32_t * ticks)\n\n/* \u8bbe\u7f6e\u95f9\u949f\uff0c\u5f53\u524dCSK6\u4ec5\u652f\u6301channel id \u4e3a0 */\ncounter_set_channel_alarm(rtc, 0, &alarm_cfg);\n")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u521b\u5efa\u7ebf\u7a0b")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-c"},"int pri = k_thread_priority_get(k_current_get());\nk_thread_create(&rtc_thread_data, rtc_stack_area,\n                K_THREAD_STACK_SIZEOF(rtc_stack_area),\n                rtc_thread, NULL, NULL, NULL, pri, 0, K_NO_WAIT);\n")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u8bbe\u7f6e\u4e00\u4e2a2S\u7684\u95f9\u949f\uff1a"),"  "),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-c"},'/*\u8bbe\u7f6e\u4e2d\u65ad\u56de\u8c03\u5904\u7406*/\nstatic void sec_counter_callback(const struct device *dev,\n                 uint8_t id,\n                 uint32_t ticks,\n                 void *ud)\n{\n    printk("Counter alarm callback at %u ms, id %d, ticks %u, ud %p\\n",\n           k_uptime_get_32(), id, ticks, ud);\n}\n\n/*\u95f9\u949f\u4e1a\u52a1\u903b\u8f91\u5b9e\u73b0*/\nvoid rtc_thread(void *v1, void *v2, void *v3){\n\n    /* \u83b7\u53d6RTC\u8bbe\u5907\u5b9e\u4f8b */\n    const struct device *rtc = NULL;\n    rtc = DEVICE_DT_GET(DT_NODELABEL(rtc1));\n    ...\n    uint32_t now;\n    uint32_t delay_time = 2;/*\u5355\u4f4d/S*/\n    struct counter_alarm_cfg alarm_cfg;\n    alarm_cfg.flags = 0;\n    alarm_cfg.callback = sec_counter_callback;\n    /* \u5728\u95f9\u949f\u5f00\u59cb\u524d\u83b7\u53d6\u5f53\u524d\u8ba1\u6570\u5668\u503c */\n    counter_get_value(rtc, &now);\n    alarm_cfg.ticks = now + delay_time;  /* \u95f9\u949f\u65f6\u95f4\u8bbe\u7f6e\u4e3a2S\uff0c\u4ece\u5f53\u524d\u8ba1\u6570\u5668\u503c\u5f00\u59cb */      \n    alarm_cfg.user_data = &alarm_cfg;\n\n    /* \u5728\u95f9\u949f\u5f00\u59cb\u524d\u5f53\u524d\u8ba1\u6570\u5668\u503c */\n    counter_get_value(rtc, &now);\n    printk("get RTC ticks value befor alarm start: %ds\\n", ticks);\n\n    /* \u8bbe\u7f6e\u95f9\u949f\uff0c\u5f53\u524dCSK6\u4ec5\u652f\u6301channel id \u4e3a0 */   \n    counter_set_channel_alarm(rtc, 0, &alarm_cfg);\n\n    /* \u5728\u95f9\u949f\u7ed3\u675f\u540e\u83b7\u53d6\u5f53\u524d\u8ba1\u6570\u5668\u503c */\n    k_msleep(3000);\n    counter_get_value(rtc, &ticks);\n    printk("RTC value: %ds\\n", ticks);\n}\n')),(0,l.kt)("h2",{id:"\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u7f16\u8bd1"),"  ")),(0,l.kt)("p",null,"\u5728app\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4ee5\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u70e7\u5f55"),"     ")),(0,l.kt)("p",null,(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u901a\u8fc7USB\u8fde\u63a5PC\uff0c\u901a\u8fc7\u70e7\u5f55\u6307\u5b8c\u6210\u70e7\u5f55\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep flash\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u67e5\u770b\u7ed3\u679c"),"  ")),(0,l.kt)("p",null,"\u53ef\u901a\u8fc7lisa\u63d0\u4f9b\u7684",(0,l.kt)("inlineCode",{parentName:"p"},"lisa term"),"\u547d\u4ee4\u67e5\u770b\u65e5\u5fd7\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa term\n")),(0,l.kt)("p",null,"\u6216\u8005\u5c06",(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u7684\u65e5\u5fd7\u4e32\u53e3",(0,l.kt)("inlineCode",{parentName:"p"},"A03 TX A02 RX"),"\u63a5\u4e32\u53e3\u677f\u8fde\u63a5\u7535\u8111\uff0c\u5728\u7535\u8111\u7aef\u4f7f\u7528\u4e32\u53e3\u8c03\u8bd5\u52a9\u624b\u67e5\u770b\u65e5\u5fd7\uff0c\u6ce2\u7279\u7387\u4e3a115200\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"*** Booting Zephyr OS build fd83997719ed  ***\nRTC name: RTC_1\nRTC top value is 4296508s\n======RTC device alarm start======\nget current counter value: 0s\nCounter alarm callback at 2385 ms, id 0, ticks 2, ud 0x81820\nget counter value after alarm end: 2s\n======RTC device alarm end======\n\n")),(0,l.kt)("p",null," \u4ece\u65e5\u5fd7\u53ef\u4ee5\u770b\u5230\uff0c\u95f9\u949f\u57282S\u540e\u89e6\u53d1\u4e86\u4e2d\u65ad\uff0c\u7b26\u5408\u793a\u4f8b\u5b9e\u73b0\u7684\u9884\u671f\uff0c\u4ee5\u4e0a\u5c31\u662f\u672c\u7ae0\u8282\u6240\u63d0\u4f9b\u7684RTC\u4f7f\u7528\u793a\u4f8b\u3002"))}d.isMDXComponent=!0},89072:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_create01-4190be564317cfb86204637eaa81c2cb.png"}}]);