(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[2412],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return s},kt:function(){return u}});var a=t(67294);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function r(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function p(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?r(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function l(e,n){if(null==e)return{};var t,a,i=function(e,n){if(null==e)return{};var t,a,i={},r=Object.keys(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||(i[t]=e[t]);return i}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var o=a.createContext({}),d=function(e){var n=a.useContext(o),t=n;return e&&(t="function"==typeof e?e(n):p(p({},n),e)),t},s=function(e){var n=d(e.components);return a.createElement(o.Provider,{value:n},e.children)},c={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},m=a.forwardRef((function(e,n){var t=e.components,i=e.mdxType,r=e.originalType,o=e.parentName,s=l(e,["components","mdxType","originalType","parentName"]),m=d(t),u=i,k=m["".concat(o,".").concat(u)]||m[u]||c[u]||r;return t?a.createElement(k,p(p({ref:n},s),{},{components:t})):a.createElement(k,p({ref:n},s))}));function u(e,n){var t=arguments,i=n&&n.mdxType;if("string"==typeof e||i){var r=t.length,p=new Array(r);p[0]=m;var l={};for(var o in n)hasOwnProperty.call(n,o)&&(l[o]=n[o]);l.originalType=e,l.mdxType="string"==typeof e?e:i,p[1]=l;for(var d=2;d<r;d++)p[d]=t[d];return a.createElement.apply(null,p)}return a.createElement.apply(null,t)}m.displayName="MDXCreateElement"},37835:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return l},contentTitle:function(){return o},metadata:function(){return d},toc:function(){return s},default:function(){return m}});var a=t(22122),i=t(19756),r=(t(67294),t(3905)),p=["components"],l={},o="GPIO",d={unversionedId:"chips/600X/application/peripheral/samples/gpio",id:"chips/600X/application/peripheral/samples/gpio",isDocsHomePage:!1,title:"GPIO",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/peripheral/samples/gpio.md",sourceDirName:"chips/600X/application/peripheral/samples",slug:"/chips/600X/application/peripheral/samples/gpio",permalink:"/chips/600X/application/peripheral/samples/gpio",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"\u5916\u8bbe\u9a71\u52a8\u7684\u4f7f\u7528",permalink:"/chips/600X/application/peripheral/overview"},next:{title:"PWM",permalink:"/chips/600X/application/peripheral/samples/pwm"}},s=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u51c6\u5907\u5de5\u4f5c",id:"\u51c6\u5907\u5de5\u4f5c",children:[]},{value:"\u521b\u5efa\u9879\u76ee",id:"\u521b\u5efa\u9879\u76ee",children:[]},{value:"\u5b9e\u73b0\u8fc7\u7a0b",id:"\u5b9e\u73b0\u8fc7\u7a0b",children:[{value:"\u7ec4\u4ef6\u914d\u7f6e",id:"\u7ec4\u4ef6\u914d\u7f6e",children:[]},{value:"\u65b9\u5f0f1\uff1a\u901a\u8fc7LED0\u8bbe\u5907\u6811\u914d\u7f6e\u5b8c\u6210LED\u63a7\u5236",id:"\u65b9\u5f0f1\uff1a\u901a\u8fc7led0\u8bbe\u5907\u6811\u914d\u7f6e\u5b8c\u6210led\u63a7\u5236",children:[]},{value:"\u65b9\u5f0f2\uff1a\u76f4\u63a5\u64cd\u4f5cGPIOA_05",id:"\u65b9\u5f0f2\uff1a\u76f4\u63a5\u64cd\u4f5cgpioa_05",children:[]}]},{value:"\u7f16\u8bd1\u548c\u70e7\u5f55",id:"\u7f16\u8bd1\u548c\u70e7\u5f55",children:[]}],c={toc:s};function m(e){var n=e.components,l=(0,i.Z)(e,p);return(0,r.kt)("wrapper",(0,a.Z)({},c,l,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"gpio"},"GPIO"),(0,r.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,r.kt)("p",null,"GPIO\u7684\u4f7f\u7528\u4f7f\u6211\u4eec\u6700\u5e38\u7528\u7684\u5916\u8bbe\u64cd\u4f5c\u4e4b\u4e00\uff0c\u672c\u7ae0\u8282\u5c06\u901a\u8fc7\u4e00\u4e2a\u7b80\u5355\u7684\u5e94\u7528\u7a0b\u5e8fBlinky\uff0c\u5b9e\u73b0GPIO\u63a7\u5236LED\u706f\u7684\u4eae\u706d\uff0c",(0,r.kt)("inlineCode",{parentName:"p"},"CSK6 SDK"),"\u63d0\u4f9b\u4e86\u57fa\u7840\u7684\u5b9e\u73b0\u793a\u4f8b\uff0c\u672c\u7ae0\u8282\u6559\u4f60\u5982\u4f55\u57fa\u4e8e",(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u5b9e\u73b0LED\u706f\u7684\u95ea\u70c1\u3002"),(0,r.kt)("h2",{id:"\u51c6\u5907\u5de5\u4f5c"},"\u51c6\u5907\u5de5\u4f5c"),(0,r.kt)("p",null,"\u5b9e\u73b0Blinky\u793a\u4f8b\u7684\u9884\u671f\u6548\u679c\u9700\u8981\u786c\u4ef6\u5f00\u53d1\u677f\u4e0a\u5fc5\u987b\u6709\u4e00\u4e2aGPIO\u8fde\u63a5\u4e86\u4e00\u4e2aLED\u706f\uff0c\u5728",(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u4e0a\u662f\u6709\u8fd9\u4e2a\u8bbe\u8ba1\u7684\uff0c\u901a\u8fc7\u67e5\u770b\u5f00\u53d1\u677f\u5e95\u677f\u539f\u7406\u56fe\uff0c\u4f60\u53ef\u4ee5\u770b\u5230LED\u5bf9\u5e94\u7684\u7535\u8def\u8bbe\u8ba1\u5982\u4e0b\u56fe\u6240\u793a\uff0c\u6211\u4eec\u53ef\u4ee5\u770b\u5230LED1(Green)\u5bf9\u5e94\u7684\u63a7\u5236\u5f15\u811a\u4e3a:GPIOA_05\n",(0,r.kt)("img",{src:t(19545).Z})),(0,r.kt)("p",null,"\u672c\u6587\u5c06\u4ecb\u7ecd\u4e24\u79cd\u57fa\u4e8e",(0,r.kt)("inlineCode",{parentName:"p"},"CSK6 SDK"),"\u7684GPIO\u63a7\u5236\u65b9\u5f0f\uff1a"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"\u65b9\u5f0f1\uff1a\u901a\u8fc7led0\u8bbe\u5907\u6811\u914d\u7f6e\u5b8c\u6210LED\u7684\u914d\u7f6e\uff0c\u5e76\u5728sample\u4e2d\u83b7\u53d6led0\u7684\u914d\u7f6e\uff0c\u5b8c\u6210\u5bf9LED\u7684\u63a7\u5236"),(0,r.kt)("li",{parentName:"ul"},"\u65b9\u5f0f2\uff1a\u7ed5\u8fc7\u8bbe\u5907\u6811\u5bf9\u7684LED\u914d\u7f6e\uff0c\u76f4\u63a5\u64cd\u4f5c\u5bf9\u5e94GPIO\u5f15\u811a\u8f93\u51fa\u9ad8\u4f4e\u7535\u5e73\uff0c\u5b8c\u6210\u5bf9LED\u7684\u63a7\u5236")),(0,r.kt)("h2",{id:"\u521b\u5efa\u9879\u76ee"},"\u521b\u5efa\u9879\u76ee"),(0,r.kt)("p",null,(0,r.kt)("inlineCode",{parentName:"p"},"CSK6 SDK"),"\u63d0\u4f9b\u4e86Blinky\u7684sample\uff0c\u4f60\u53ef\u4ee5\u901a\u8fc7\u4e00\u4e0b\u6307\u4ee4\u521b\u5efa\u4e00\u4e2aBlinky\u9879\u76ee\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,r.kt)("p",null,(0,r.kt)("img",{src:t(91162).Z}),"\n",(0,r.kt)("img",{src:t(14817).Z}),"\n",(0,r.kt)("img",{src:t(79702).Z}),"\n",(0,r.kt)("img",{src:t(33293).Z}),"\nBlinky sample\u521b\u5efa\u6210\u529f\u3002"),(0,r.kt)("h2",{id:"\u5b9e\u73b0\u8fc7\u7a0b"},"\u5b9e\u73b0\u8fc7\u7a0b"),(0,r.kt)("h3",{id:"\u7ec4\u4ef6\u914d\u7f6e"},"\u7ec4\u4ef6\u914d\u7f6e"),(0,r.kt)("p",null,"\u5728prj.conf\u6587\u4ef6\u4e2d\u6dfb\u52a0\u9879\u76ee\u57fa\u7840\u7ec4\u4ef6\u914d\u7f6e\u914d\u7f6e:"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"...\nCONFIG_GPIO=y\n")),(0,r.kt)("h3",{id:"\u65b9\u5f0f1\uff1a\u901a\u8fc7led0\u8bbe\u5907\u6811\u914d\u7f6e\u5b8c\u6210led\u63a7\u5236"},"\u65b9\u5f0f1\uff1a\u901a\u8fc7LED0\u8bbe\u5907\u6811\u914d\u7f6e\u5b8c\u6210LED\u63a7\u5236"),(0,r.kt)("h4",{id:"\u8bbe\u5907\u6811\u914d\u7f6e"},"\u8bbe\u5907\u6811\u914d\u7f6e"),(0,r.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano.dts"),"\u662f\u6211\u4eec\u5f53\u524d\u4f7f\u7528\u5f00\u53d1\u677f\u7684\u8bbe\u5907\u6811\u6587\u4ef6\uff0c\u4f4d\u4e8eSDK\u76ee\u5f55\u7684 ",(0,r.kt)("inlineCode",{parentName:"p"},"zephyr/board/arm/csk6002_9s_nano"),"\u6587\u4ef6\u5939\u4e2d\u3002"))),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"\u9996\u5148\u6211\u4eec\u9700\u8981\u5728\u5f53\u524dboad\u7684\u8bbe\u5907\u6811\u4e2d\u6dfb\u52a0led0\u7684\u914d\u7f6e\uff0c\u5373\u5728",(0,r.kt)("inlineCode",{parentName:"li"},"csk6002_9s_nano.dts"),"\u4e2d\u5b9e\u73b0",(0,r.kt)("inlineCode",{parentName:"li"},"led0"),"\u548c",(0,r.kt)("inlineCode",{parentName:"li"},"GPIOA_05"),"\u7684\u5339\u914d,\u914d\u7f6e\u5982\u4e0b:")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},'/ {\n        model = "csk6002 9s nano";\n        compatible = "csk,csk6002_9s_nano";\n    aliases {\n            led0 = &board_led_2;\n            ...\n    };\n    ...\n    leds {\n            compatible = "gpio-leds";\n            board_led_2: board_led_2 {\n                    gpios = <&gpioa 5 0>;\n                    label = "User BOARD_LED_2";\n            };\n    };\n    ...\n};\n')),(0,r.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},"\u5728",(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u8fd9\u4e2aboard\u4e2d\u7684dts\u6587\u4ef6\u4e2d\u5df2\u7ecf\u9ed8\u8ba4\u6dfb\u52a0\u4e86",(0,r.kt)("inlineCode",{parentName:"p"},"led0"),"\u7684\u914d\u7f6e\uff0c\u6240\u4ee5\u6211\u4eec\u65e0\u9700\u518d\u589e\u52a0\uff0c\u5f53\u4f60\u5728\u9002\u914d\u4e00\u4e2a\u65b0\u7684\u677f\u5b50\u521b\u5efa\u65b0\u7684board\u65f6\uff0c\u9700\u8981\u786e\u4fddled0\u7684\u914d\u7f6e\u662f\u5b58\u5728\u7684\u3002"))),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"\u5176\u6b21\uff0c\u5728sample\u4ee3\u7801\u4e2d\u83b7\u53d6\u8bbe\u5907\u6811led0\u914d\u7f6e\u4fe1\u606f")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},'#include <zephyr.h>\n#include <device.h>\n#include <devicetree.h>\n#include <drivers/gpio.h>\n...\n/* The devicetree node identifier for the "led0" alias. */\n#define LED0_NODE DT_ALIAS(led0)\n\n#if DT_NODE_HAS_STATUS(LED0_NODE, okay)\n#define LED0    DT_GPIO_LABEL(LED0_NODE, gpios)\n#define PIN DT_GPIO_PIN(LED0_NODE, gpios)\n#define FLAGS   DT_GPIO_FLAGS(LED0_NODE, gpios)\n\n')),(0,r.kt)("p",null,"\u5982\u679c\u4f60\u7684board\u672a\u5b9e\u73b0",(0,r.kt)("inlineCode",{parentName:"p"},"led0"),"\u7684\u914d\u7f6e\uff0c\u90a3\u4e48\u5728\u7f16\u8bd1sample\u65f6\u4f1a\u9047\u5230\u5982\u4e0b\u9519\u8bef\u63d0\u793a\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"Unsupported board: led0 devicetree alias is not defined\n")),(0,r.kt)("h4",{id:"\u83b7\u53d6gpio\u8bbe\u5907\u5b9e\u4f8b\u5e76\u5b8c\u6210led\u63a7\u5236\u903b\u8f91"},"\u83b7\u53d6GPIO\u8bbe\u5907\u5b9e\u4f8b\u5e76\u5b8c\u6210LED\u63a7\u5236\u903b\u8f91"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"void main(void)\n{\n...\nconst struct device *dev;\nbool led_is_on = true;\n\ndev = device_get_binding(LED0);\n\nret = gpio_pin_configure(dev, PIN, GPIO_OUTPUT_ACTIVE | FLAGS);\n\nwhile (1) {\n        gpio_pin_set(dev, PIN, (int)led_is_on);\n        led_is_on = !led_is_on;\n        k_msleep(SLEEP_TIME_MS);\n    }\n}\n")),(0,r.kt)("h3",{id:"\u65b9\u5f0f2\uff1a\u76f4\u63a5\u64cd\u4f5cgpioa_05"},"\u65b9\u5f0f2\uff1a\u76f4\u63a5\u64cd\u4f5cGPIOA_05"),(0,r.kt)("p",null,"\u57fa\u4e8e",(0,r.kt)("inlineCode",{parentName:"p"},"Zephyr-CSK-SDK"),"\uff0c\u6211\u4eec\u53ef\u4ee5\u5728sample\u4e2d\u83b7\u53d6",(0,r.kt)("inlineCode",{parentName:"p"},"GPIOA_05"),"\u5b9e\u4f8b\uff0c\u5e76\u5b8c\u6210IO\u7535\u5e73\u63a7\u5236\uff0c\u5b9e\u73b0\u65b9\u5f0f\u5982\u4e0b\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"#define PIN 5\n#define FLAGS 0\n\nvoid main(void)\n{\n    const struct device *dev;\n    bool led_is_on = true;\n\n    dev = DEVICE_DT_GET(DT_NODELABEL(gpioa));\n    ...\n    ret = gpio_pin_configure(dev, PIN, GPIO_OUTPUT_ACTIVE | FLAGS); \n    ...\n    while (1) {\n        gpio_pin_set(dev, PIN, (int)led_is_on);\n        led_is_on = !led_is_on;\n        k_msleep(SLEEP_TIME_MS);\n    }\n}\n")),(0,r.kt)("p",null,"\u4ee3\u7801\u5b9e\u73b0\u4e2d\u4f7f\u7528\u5230\u4e86",(0,r.kt)("inlineCode",{parentName:"p"},"DEVICE_DT_GET"),"\u8fd9\u4e2a\u63a5\u53e3\u83b7\u53d6GPIO\u5b9e\u4f8b\uff0c\u5982\u679c\u4f60\u5bf9\u8fd9\u4e2a\u63a5\u53e3\u5f88\u964c\u751f\uff0c\u8fd9\u91cc\u7ed9\u4f60\u4e00\u4e9b\u63d0\u793a\uff0c\u5982\u679c\u4f60\u5df2\u7ecf\u5f88\u719f\u6089\uff0c\u90a3\u4e48\u4e0b\u9762\u8fd9\u6bb5\u8bf4\u660e\u53ef\u4ee5\u8df3\u8fc7\u3002\n",(0,r.kt)("inlineCode",{parentName:"p"},"DEVICE_DT_GET"),"\uff1a\u901a\u8fc7",(0,r.kt)("inlineCode",{parentName:"p"},"node_id"),"\u83b7\u53d6\u6307\u5411\u8bbe\u5907\u5bf9\u8c61\u7684\u6307\u9488\uff0c\u6211\u4eec\u53ef\u4ee5\u5728",(0,r.kt)("inlineCode",{parentName:"p"},"zephyr API"),"\u4e2d\u627e\u5230\u5bf9\u5e94\u7684\u63cf\u8ff0\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"DEVICE_DT_GET(node_id)\n\nObtain a pointer to a device object by node_id.\nReturn the address of a device object created by DEVICE_DT_INIT(), using the dev_name derived from node_id\n\nParameters\nnode_id \u2013 The same as node_id provided to DEVICE_DT_DEFINE()\n\nReturns\nA pointer to the device object created by DEVICE_DT_DEFINE()\n")),(0,r.kt)("p",null,"\u90a3\u4e48\u5982\u4f55\u83b7\u53d6\u5bf9\u5e94\u7684node_id\u5462\uff1f\u7b54\u6848\u662f\u901a\u8fc7label\u83b7\u53d6node_id\uff1a",(0,r.kt)("inlineCode",{parentName:"p"},"DT_NODELABEL(gpioa)"),"\u3002\n",(0,r.kt)("inlineCode",{parentName:"p"},"gpioa"),"\u7684label\u5728",(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano.dts"),"\u7684\u914d\u7f6e\u53ef\u4ee5\u627e\u5230\uff0c\u5f53\u7136\u8fd8\u6709gpiob\u7684",(0,r.kt)("inlineCode",{parentName:"p"},"label"),"\u914d\u7f6e\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},'...\n&gpioa {\n        status = "okay";\n};\n\n&gpiob {\n        status = "okay";\n};\n...\n')),(0,r.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},(0,r.kt)("inlineCode",{parentName:"p"},'status = "okay"'),"\u8868\u793a\u652f\u6301gpio\u7684\u63a7\u5236\uff0c",(0,r.kt)("inlineCode",{parentName:"p"},'status = "disabled"'),"\u8868\u793a\u4e0d\u652f\u6301\u3002"))),(0,r.kt)("h2",{id:"\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"\u7f16\u8bd1"),"\n\u5728sample\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4e00\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,r.kt)("p",null,"\u7f16\u8bd1\u6210\u529f\uff1a\n",(0,r.kt)("img",{src:t(68890).Z})),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"\u70e7\u5f55"),(0,r.kt)("inlineCode",{parentName:"li"},"csk6002_9s_nano"),"\u901a\u8fc7USB\u8fde\u63a5PC\uff0c\u901a\u8fc7\u70e7\u5f55\u6307\u4ee4\u5f00\u59cb\u70e7\u5f55\uff1a")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep flash --runner pyocd\n")),(0,r.kt)("p",null,"\u70e7\u5f55\u6210\u529f\uff1a\n",(0,r.kt)("img",{src:t(73405).Z})),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("strong",{parentName:"li"},"\u6548\u679c"),"\n\u9884\u671f\u7684\u6548\u679c\u5e94\u5982\u4e0b\u4e24\u4e2a\u56fe\u7247\u6240\u793a\uff0c\u5f00\u53d1\u677f\u4e0a\u7684LED\u706f(\u7eff)\u5728\u4e0d\u65ad\u7684\u95ea\u70c1\uff0c\u5982\u679c\u5728\u4f60\u7684\u5361\u53d1\u677f\u4e0a\u5b9e\u73b0\u4e86\u8fd9\u4e2a\u6548\u679c\uff0c\u90a3\u4e48\u606d\u559c\uff0c\u4f60\u987a\u5229\u7684\u5b8c\u6210\u4e86LED\u7684\u63a7\u5236\uff0c\u5728CSK6\u7684\u5f00\u53d1\u4e0a\u53c8\u8fc8\u51fa\u4e86\u4e00\u6b65\uff01")),(0,r.kt)("p",null,(0,r.kt)("img",{src:t(22106).Z}),"\n",(0,r.kt)("img",{src:t(20250).Z})))}m.isMDXComponent=!0},68890:function(e,n,t){"use strict";n.Z=t.p+"assets/images/build-2c4effd271da8962e836b3290295b004.png"},91162:function(e,n,t){"use strict";n.Z=t.p+"assets/images/create_blinky01-96199d2abe5c9e4552b478aae9d0184d.png"},14817:function(e,n,t){"use strict";n.Z=t.p+"assets/images/create_blinky02-0cfaf6f1da01a51c2e6ec72fc2781965.png"},79702:function(e,n,t){"use strict";n.Z=t.p+"assets/images/create_blinky03-ba2396f5742d90986eb9868eda314c33.png"},33293:function(e,n,t){"use strict";n.Z=t.p+"assets/images/create_blinky04-dfaf50b3c8095c891a6608d8959f19fd.png"},73405:function(e,n,t){"use strict";n.Z=t.p+"assets/images/flash-5e8b0d6c780a73c444d93c6bcd490b1c.png"},20250:function(e,n,t){"use strict";n.Z=t.p+"assets/images/led_off-c473a6a4336d9c7cd857f8898570af23.png"},22106:function(e,n,t){"use strict";n.Z=t.p+"assets/images/led_on-e1e39fb27355d21cb6d495ea7a96dc75.png"},19545:function(e,n,t){"use strict";n.Z=t.p+"assets/images/led_pin-07a357fff1557e244a61f7e8302145d7.png"}}]);