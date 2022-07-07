(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[7985],{3905:function(e,n,t){"use strict";t.d(n,{Zo:function(){return d},kt:function(){return m}});var a=t(67294);function i(e,n,t){return n in e?Object.defineProperty(e,n,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[n]=t,e}function r(e,n){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);n&&(a=a.filter((function(n){return Object.getOwnPropertyDescriptor(e,n).enumerable}))),t.push.apply(t,a)}return t}function l(e){for(var n=1;n<arguments.length;n++){var t=null!=arguments[n]?arguments[n]:{};n%2?r(Object(t),!0).forEach((function(n){i(e,n,t[n])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):r(Object(t)).forEach((function(n){Object.defineProperty(e,n,Object.getOwnPropertyDescriptor(t,n))}))}return e}function c(e,n){if(null==e)return{};var t,a,i=function(e,n){if(null==e)return{};var t,a,i={},r=Object.keys(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||(i[t]=e[t]);return i}(e,n);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(a=0;a<r.length;a++)t=r[a],n.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(i[t]=e[t])}return i}var p=a.createContext({}),s=function(e){var n=a.useContext(p),t=n;return e&&(t="function"==typeof e?e(n):l(l({},n),e)),t},d=function(e){var n=s(e.components);return a.createElement(p.Provider,{value:n},e.children)},o={inlineCode:"code",wrapper:function(e){var n=e.children;return a.createElement(a.Fragment,{},n)}},u=a.forwardRef((function(e,n){var t=e.components,i=e.mdxType,r=e.originalType,p=e.parentName,d=c(e,["components","mdxType","originalType","parentName"]),u=s(t),m=i,_=u["".concat(p,".").concat(m)]||u[m]||o[m]||r;return t?a.createElement(_,l(l({ref:n},d),{},{components:t})):a.createElement(_,l({ref:n},d))}));function m(e,n){var t=arguments,i=n&&n.mdxType;if("string"==typeof e||i){var r=t.length,l=new Array(r);l[0]=u;var c={};for(var p in n)hasOwnProperty.call(n,p)&&(c[p]=n[p]);c.originalType=e,c.mdxType="string"==typeof e?e:i,l[1]=c;for(var s=2;s<r;s++)l[s]=t[s];return a.createElement.apply(null,l)}return a.createElement.apply(null,t)}u.displayName="MDXCreateElement"},81711:function(e,n,t){"use strict";t.r(n),t.d(n,{frontMatter:function(){return c},contentTitle:function(){return p},metadata:function(){return s},toc:function(){return d},default:function(){return u}});var a=t(22122),i=t(19756),r=(t(67294),t(3905)),l=["components"],c={},p="I2C",s={unversionedId:"chips/600X/application/peripheral/samples/I2C",id:"chips/600X/application/peripheral/samples/I2C",isDocsHomePage:!1,title:"I2C",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/peripheral/samples/I2C.md",sourceDirName:"chips/600X/application/peripheral/samples",slug:"/chips/600X/application/peripheral/samples/I2C",permalink:"/chips/600X/application/peripheral/samples/I2C",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"UART",permalink:"/chips/600X/application/peripheral/samples/UART"},next:{title:"SPI",permalink:"/chips/600X/application/peripheral/samples/spi"}},d=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"API\u63a5\u53e3",id:"api\u63a5\u53e3",children:[]},{value:"\u4f7f\u7528\u793a\u4f8b",id:"\u4f7f\u7528\u793a\u4f8b",children:[{value:"\u51c6\u5907\u5de5\u4f5c",id:"\u51c6\u5907\u5de5\u4f5c",children:[]},{value:"\u83b7\u53d6sample\u9879\u76ee",id:"\u83b7\u53d6sample\u9879\u76ee",children:[]},{value:"\u7ec4\u4ef6\u914d\u7f6e",id:"\u7ec4\u4ef6\u914d\u7f6e",children:[]},{value:"I2C\u8bbe\u5907\u6811\u914d\u7f6e",id:"i2c\u8bbe\u5907\u6811\u914d\u7f6e",children:[]},{value:"\u5e94\u7528\u903b\u8f91",id:"\u5e94\u7528\u903b\u8f91",children:[]},{value:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0",id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0",children:[]},{value:"\u7f16\u8bd1\u548c\u70e7\u5f55",id:"\u7f16\u8bd1\u548c\u70e7\u5f55",children:[]}]}],o={toc:d};function u(e){var n=e.components,c=(0,i.Z)(e,l);return(0,r.kt)("wrapper",(0,a.Z)({},o,c,{components:n,mdxType:"MDXLayout"}),(0,r.kt)("h1",{id:"i2c"},"I2C"),(0,r.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,r.kt)("p",null,"I2C\u662f\u6211\u4eec\u5e38\u7528\u7684\u5916\u8bbe\u529f\u80fd\u4e4b\u4e00\uff0cCSK6 SDK I2C\u5916\u8bbe\u9a71\u52a8\u91c7\u7528\u6807\u51c6\u7684I2C\u603b\u7ebf\u534f\u8bae\uff0c\u672c\u8282\u6211\u4eec\u5c06\u901a\u8fc7\u793a\u4f8b\u8bb2\u89e3\u5982\u4f55\u4f7f\u7528CSK6 SDK\u7684I2C API\u63a5\u53e3\u5b9e\u73b0\u6570\u636e\u7684\u6536\u53d1\u3002"),(0,r.kt)("p",null,"CSK6 \u82af\u7247\u6709\u4e24\u4e2aI2C \u786c\u4ef6\u5916\u8bbe\u3002\nCSK6 I2C\u9a71\u52a8\u529f\u80fd\u7279\u6027\u5982\u4e0b\uff1a"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301\u4e3b\u8bbe\u5907\u6216\u4ece\u8bbe\u5907\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301\u4e09\u79cd\u53ef\u9009\u4f20\u8f93\u901f\u5ea6\u6a21\u5f0f Standard-mode (100 kb/s)\u3001Fast-mode (400 kb/s) \u548cFast-mode plus (1 Mb/s)\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301DMA\u4f20\u9001\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301\u4e3b\u53d1\u3001\u4e3b\u6536\u3001\u4ece\u53d1\u3001\u4ece\u6536\u6a21\u5f0f\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301\u591a\u4e3b\u673a\u6a21\u5f0f\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301\u5730\u5740\u53c2\u6570\u4e3a 7-bit\u548c10-bit\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301general call\u6a21\u5f0f\u3002"),(0,r.kt)("li",{parentName:"ul"},"\u652f\u6301\u65f6\u949f\u5ef6\u5c55\u3002")),(0,r.kt)("h2",{id:"api\u63a5\u53e3"},"API\u63a5\u53e3"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-c"},"/*i2c\u63a7\u5236\u5668\u914d\u7f6e*/\ni2c_configure()\n/*i2c\u5199\u6570\u636e*/\ni2c_write()\n/*i2c\u8bfb\u6570\u636e*/\ni2c_read()\n")),(0,r.kt)("p",null,"\u66f4\u591aI2C API\u63a5\u53e3\u8bf7\u8bbf\u95eezephyr\u5b98\u7f51",(0,r.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__i2c__interface.html"},"I2C Interface"),"\u3002"),(0,r.kt)("h2",{id:"\u4f7f\u7528\u793a\u4f8b"},"\u4f7f\u7528\u793a\u4f8b"),(0,r.kt)("h3",{id:"\u51c6\u5907\u5de5\u4f5c"},"\u51c6\u5907\u5de5\u4f5c"),(0,r.kt)("p",null,"\u672c\u793a\u4f8b\u57fa\u4e8e ",(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u5b9e\u73b0\uff0c\u4f7f\u7528\u4e86\u4e24\u7ec4I2C\u4f5c\u4e3a\u4e3b\u4ece\u8bbe\u5907\u8fdb\u884c\u6570\u636e\u7684\u901a\u8baf\uff0c\u5176\u4e2d"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"i2c0(GPIO_A_04, GPIO_A_05)"),"\u4f5c\u4e3a\u4e3b\u8bbe\u5907(master)"),(0,r.kt)("li",{parentName:"ul"},(0,r.kt)("inlineCode",{parentName:"li"},"i2c1(GPIO_A_06, GPIO_A_07)"),"\u4f5c\u4e3a\u4ece\u8bbe\u5907(slave)\n\u4f7f\u7528\u675c\u90a6\u7ebf\u5c06",(0,r.kt)("inlineCode",{parentName:"li"},"i2c0(GPIO_A_04, GPIO_A_05)"),"\u548c",(0,r.kt)("inlineCode",{parentName:"li"},"i2c1(GPIO_A_06, GPIO_A_07)"),"\u901a\u8fde\u63a5\uff0c\u5982\u4e0b\u56fe\u793a\uff1a")),(0,r.kt)("p",null,(0,r.kt)("img",{src:t(66960).Z})),(0,r.kt)("div",{className:"admonition admonition-tip alert alert--success"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"12",height:"16",viewBox:"0 0 12 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.5 0C3.48 0 1 2.19 1 5c0 .92.55 2.25 1 3 1.34 2.25 1.78 2.78 2 4v1h5v-1c.22-1.22.66-1.75 2-4 .45-.75 1-2.08 1-3 0-2.81-2.48-5-5.5-5zm3.64 7.48c-.25.44-.47.8-.67 1.11-.86 1.41-1.25 2.06-1.45 3.23-.02.05-.02.11-.02.17H5c0-.06 0-.13-.02-.17-.2-1.17-.59-1.83-1.45-3.23-.2-.31-.42-.67-.67-1.11C2.44 6.78 2 5.65 2 5c0-2.2 2.02-4 4.5-4 1.22 0 2.36.42 3.22 1.19C10.55 2.94 11 3.94 11 5c0 .66-.44 1.78-.86 2.48zM4 14h5c-.23 1.14-1.3 2-2.5 2s-2.27-.86-2.5-2z"}))),"tip")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},"\u901a\u5e38I2C\u901a\u8bafSCL\u548cSDA\u4e4b\u95f4\u90fd\u9700\u8981\u63a5\u4e0a\u62c9\u7535\u963b(\u5927\u5c0f\u7531\u901f\u5ea6\u548c\u5bb9\u6027\u8d1f\u8f7d\u51b3\u5b9a\u4e00\u822c\u57283.3K-10K\u4e4b\u95f4)\u4ee5\u4fdd\u8bc1\u6570\u636e\u7684\u7a33\u5b9a\u6027\uff0c\u51cf\u5c11\u5e72\u6270\uff0c\u5982\u4e0b\u56fe\u793a:"))),(0,r.kt)("p",null,(0,r.kt)("img",{src:t(41612).Z})),(0,r.kt)("h3",{id:"\u83b7\u53d6sample\u9879\u76ee"},"\u83b7\u53d6sample\u9879\u76ee"),(0,r.kt)("p",null,"\u901a\u8fc7Lisa\u547d\u4ee4\u521b\u5efa\u9879\u76ee\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,r.kt)("p",null,(0,r.kt)("img",{src:t(46425).Z}),"\n",(0,r.kt)("img",{src:t(3584).Z}),"\n",(0,r.kt)("img",{src:t(2026).Z}),"\n",(0,r.kt)("img",{src:t(49682).Z})),(0,r.kt)("h3",{id:"\u7ec4\u4ef6\u914d\u7f6e"},"\u7ec4\u4ef6\u914d\u7f6e"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"CONFIG_STDOUT_CONSOLE=y\nCONFIG_PRINTK=y\n# I2C\u914d\u7f6e\nCONFIG_I2C=y\n")),(0,r.kt)("h3",{id:"i2c\u8bbe\u5907\u6811\u914d\u7f6e"},"I2C\u8bbe\u5907\u6811\u914d\u7f6e"),(0,r.kt)("p",null,(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u63d0\u4f9b\u4e86\u591a\u7ec4I2C\u3002\u672c\u793a\u4f8b\u4f7f\u7528",(0,r.kt)("inlineCode",{parentName:"p"},"i2c0(GPIO_A_04, GPIO_A_05)"),"\u548c",(0,r.kt)("inlineCode",{parentName:"p"},"i2c1(GPIO_A_06, GPIO_A_07)"),"\u4e24\u7ec4GPIO\u53e3\uff0c\u56e0\u6b64\u9700\u8981\u5728\u8bbe\u5907\u6811\u4e2d\u5c06\u8fd9\u4e24\u7ec4GPIO\u590d\u7528\u4e3aI2C\u5f15\u811a\u529f\u80fd\uff0c\u53ef\u901a\u8fc7",(0,r.kt)("inlineCode",{parentName:"p"},"boad overlay"),"\u7684\u65b9\u5f0f\u5b8c\u6210I2C\u5f15\u811a\u7684\u914d\u7f6e\u3002"),(0,r.kt)("p",null,"\u5728app\u76ee\u5f55\u4e0b\u589e\u52a0",(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano.overlay"),"\u6587\u4ef6\u5e76\u6dfb\u52a0\u5982\u4e0b\u914d\u7f6e\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-c"},'/*\n * SPDX-License-Identifier: Apache-2.0\n *\n * Copyright (c) 2022 Listenai\n */\n\n&csk6002_9s_nano_pinctrl{\n                /* \u5c06GPIOA_4\u914d\u7f6e\u4e3ai2c0 scl*/\n                pinctrl_i2c0_scl_default: i2c0_scl_default{\n                /* \u5c06pa4 \u8bbe\u7f6efunction\u4e3a8 \u590d\u7528I2C\u5f15\u811a\u529f\u80fd */\n                    pinctrls = <&pinmuxa 4 8>;\n                };\n                /* \u5c06GPIOA_5 i2c0 sda*/\n                pinctrl_i2c0_sda_default: i2c0_sda_default{\n                     /* \u5c06pa5 \u8bbe\u7f6efunction\u4e3a8 \u590d\u7528I2C\u5f15\u811a\u529f\u80fd */\n                    pinctrls = <&pinmuxa 5 8>;\n                };\n                pinctrl_i2c1_scl_default: i2c1_scl_default{\n                    pinctrls = <&pinmuxa 7 9>;\n                };\n                pinctrl_i2c1_sda_default: i2c1_sda_default{\n                    pinctrls = <&pinmuxa 6 9>;\n                };\n\n};\n\n&i2c1 {\n        status = "okay";\n        pinctrl-0 = <&pinctrl_i2c1_scl_default &pinctrl_i2c1_sda_default>; \n        pinctrl-names = "default";\n};\n\n&i2c0 {\n        status = "okay";\n        pinctrl-0 = <&pinctrl_i2c0_scl_default &pinctrl_i2c0_sda_default>; \n        pinctrl-names = "default";\n};\n')),(0,r.kt)("div",{className:"admonition admonition-note alert alert--secondary"},(0,r.kt)("div",{parentName:"div",className:"admonition-heading"},(0,r.kt)("h5",{parentName:"div"},(0,r.kt)("span",{parentName:"h5",className:"admonition-icon"},(0,r.kt)("svg",{parentName:"span",xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"},(0,r.kt)("path",{parentName:"svg",fillRule:"evenodd",d:"M6.3 5.69a.942.942 0 0 1-.28-.7c0-.28.09-.52.28-.7.19-.18.42-.28.7-.28.28 0 .52.09.7.28.18.19.28.42.28.7 0 .28-.09.52-.28.7a1 1 0 0 1-.7.3c-.28 0-.52-.11-.7-.3zM8 7.99c-.02-.25-.11-.48-.31-.69-.2-.19-.42-.3-.69-.31H6c-.27.02-.48.13-.69.31-.2.2-.3.44-.31.69h1v3c.02.27.11.5.31.69.2.2.42.31.69.31h1c.27 0 .48-.11.69-.31.2-.19.3-.42.31-.69H8V7.98v.01zM7 2.3c-3.14 0-5.7 2.54-5.7 5.68 0 3.14 2.56 5.7 5.7 5.7s5.7-2.55 5.7-5.7c0-3.15-2.56-5.69-5.7-5.69v.01zM7 .98c3.86 0 7 3.14 7 7s-3.14 7-7 7-7-3.12-7-7 3.14-7 7-7z"}))),"note")),(0,r.kt)("div",{parentName:"div",className:"admonition-content"},(0,r.kt)("p",{parentName:"div"},"\u5982\u679c\u60a8\u60f3\u4e86\u89e3\u66f4\u591a\u5173\u4e8e\u8bbe\u5907\u6811\u7684\u4fe1\u606f\uff0c\u8bf7\u5b66\u4e60",(0,r.kt)("a",{parentName:"p",href:"/chips/600X/application/device_tree"},"\u8bbe\u5907\u6811"),"\u7ae0\u8282\u3002"))),(0,r.kt)("h3",{id:"\u5e94\u7528\u903b\u8f91"},"\u5e94\u7528\u903b\u8f91"),(0,r.kt)("p",null,"\u672c\u5b9e\u4f8b\u4e2d\u4e3b\u8981\u5b9e\u73b0\u4e86\u4ee5\u4e0b\u6570\u636e\u6536\u53d1\uff1a"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"master\u4e00\u6b21\u5411slave\u53d1\u9001256\u4e2a\u5b57\u8282\u6570\u636e\uff0cslave\u63a5\u6536256\u5b57\u8282\u6570\u636e\u5e76\u901a\u8fc7\u4e32\u53e3\u6253\u5370\u8f93\u51fa"),(0,r.kt)("li",{parentName:"ul"},"slave\u4e00\u6b21\u5411master\u53d1\u9001256\u4e2a\u5b57\u8282\u6570\u636e\uff0cmaster\u63a5\u6536256\u5b57\u8282\u6570\u636e\u5e76\u901a\u8fc7\u4e32\u53e3\u6253\u5370\u8f93\u51fa"),(0,r.kt)("li",{parentName:"ul"},"master\u6bcf\u969410S\u53d1\u9001\u4e00\u6b21256\u4e2a\u5b57\u8282\u6570\u636e\u5e76\u4ee5\u6b64\u5faa\u73af\uff0cslave\u63a5\u6536\u6570\u636e\u5e76\u6253\u5370\u8f93\u51fa")),(0,r.kt)("h3",{id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0"},"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"I2C\u521d\u59cb\u5316\uff1a")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-c"},"/*\u5b9a\u4e49\u4e00\u4e2aI2C\u8bbe\u5907\u540d\u79f0\uff0c\u5e76\u901a\u8fc7\u8bbe\u5907\u6811\u63a5\u53e3\u83b7\u53d6I2C\u7684\u8bbe\u5907\u6811\u4fe1\u606f*/\n#define I2C0_DEV_NAME DT_LABEL(DT_ALIAS(i2c_0)) \n#define I2C1_DEV_NAME DT_LABEL(DT_ALIAS(i2c_1))\n\n/* \u4ece\u673a\u5730\u5740\uff1acsk6\u9ed8\u8ba4\u7684\u4ece\u673a\u5730\u5740\u4e3a:0x35*/\n#define SLAVE_ADDRESS (0x35) \n\n/* \u8bbe\u7f6eI2C\u7684\u65f6\u949f\u901f\u7387\u548c\u4e3b\u4ece\u6a21\u5f0f*/\nuint32_t i2c0_cfg = I2C_SPEED_SET(I2C_SPEED_STANDARD) | I2C_MODE_MASTER;\nuint32_t i2c1_cfg = 0;\n\n/*\u5b9a\u4e49I2C\u8bbe\u5907\u5b9e\u4f8b*/\nconst struct device *i2c0_dev = NULL;\nconst struct device *i2c1_dev = NULL;\n\n/*\u5b9a\u4e49\u7528\u4e8eI2C\u4e3b\u4ece\u8bbe\u5907\u6570\u636e\u6536\u53d1\u7684\u6570\u7ec4\uff0c\u957f\u5ea6\u4e3a256\u4e2a\u5b57\u8282*/\nuint8_t data0[256] = { 0 };\nuint8_t data1[256] = { 0 };\n\nvoid main(void)\n{\n    /*\u914d\u7f6eI2C\u8bbe\u5907*/\n    i2c_configure(i2c0_dev, i2c0_cfg);\n    i2c_configure(i2c1_dev, i2c1_cfg);\n\n    /*\u521b\u5efa\u4e00\u4e2a\u7ebf\u7a0b\uff0c\u7528\u4e8e\u5904\u7406\u4ece\u8bbe\u5907\u7684\u6570\u636e\u6536\u53d1*/\n    k_thread_create(&slave_thread_data, slave_stack_area,\n                    K_THREAD_STACK_SIZEOF(slave_stack_area),\n                    i2c_slave_thread, NULL, NULL, NULL, pri, 0, K_NO_WAIT);\n    /*\u521b\u5efa\u4e00\u4e2a\u7ebf\u7a0b\uff0c\u7528\u4e8e\u5904\u7406\u4e3b\u8bbe\u5907\u7684\u6570\u636e\u6536\u53d1*/\n    k_thread_create(&master_thread_data, master_stack_area,\n                    K_THREAD_STACK_SIZEOF(master_stack_area), i2c_master_thread, NULL,\n                    NULL, NULL, pri, 0, K_NO_WAIT);\n}\n")),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"master\u548cslave\u6570\u636e\u6536\u53d1\u5b9e\u73b0")),(0,r.kt)("p",null,"i2c_master_thread\u5b9e\u73b0\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-c"},"void i2c_master_thread(void *v1, void *v2, void *v3)\n{\n    /* data0\u586b\u5145\u6570\u636e */\n    for (int i = 0; i < sizeof(data0) - 1; i++) {\n        data0[i] = i;\n    }\n\n    /* 1. master\u53d1\u9001\u6570\u636e\uff0cslave\u63a5\u6536 */\n    i2c_write(i2c0_dev, data0, sizeof(data0) - 1, SLAVE_ADDRESS);//master\u5199\u6570\u636e\n    ...\n    /* 2. master\u63a5\u6536\u6570\u636e */\n    i2c_read(i2c0_dev, data0, sizeof(data0), SLAVE_ADDRESS);\n    ...\n    /* 3. master\u95f4\u969410S\u5faa\u73af\u53d1\u9001\u6570\u636e */\n    while (1) {\n        k_msleep(10000);\n        i2c_write(i2c0_dev, data0, sizeof(data0) - 1, SLAVE_ADDRESS);  \n    }\n}\n")),(0,r.kt)("p",null,"i2c_slave_thread\u5b9e\u73b0\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre",className:"language-c"},'void i2c_slave_thread(void *v1, void *v2, void *v3)\n{\n    /* master\u53d1\u9001\u6570\u636e\uff0cslave\u63a5\u6536 */\n    i2c_read(i2c1_dev, data1, sizeof(data1), 0);\n\n    /* \u901a\u8fc7\u4e32\u53e3\u6253\u5370\u6570\u636e */\n    for (int i = 0; i < sizeof(data1); i++) {\n        printk("data1[%d]: %d\\n", i, data1[i]);\n    }\n    ...\n    /* \u6e05\u7a7a\u6570\u636e\u5e76\u91cd\u65b0\u586b\u88c5 */\n    memset(data1, sizeof(data1), 0);\n    for (int i = 0; i < sizeof(data1); i++) {\n        data1[i] = 256 - i;\n    }\n    ...\n    /* 2.slave\u53d1\u9001\u6570\u636e\uff0cmaster\u63a5\u6536 */\n    i2c_write(i2c1_dev, data1, sizeof(data1) - 1, 0);\n    ...\n    /* 3. slave\u5faa\u73af\u7b49\u5f85\u6570\u636e */\n    while (1) {\n        memset(data1, sizeof(data1), 0);\n        i2c_read(i2c1_dev, data1, sizeof(data1), 0);\n    }\n}\n')),(0,r.kt)("h3",{id:"\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,r.kt)("h4",{id:"\u7f16\u8bd1"},"\u7f16\u8bd1"),(0,r.kt)("p",null,"\u5728app\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4ee5\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,r.kt)("h4",{id:"\u70e7\u5f55"},"\u70e7\u5f55"),(0,r.kt)("p",null,(0,r.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u901a\u8fc7USB\u8fde\u63a5PC\uff0c\u901a\u8fc7\u70e7\u5f55\u6307\u4ee4\u5f00\u59cb\u70e7\u5f55\uff1a"),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"lisa zep flash --runner pyocd\n")),(0,r.kt)("h4",{id:"\u67e5\u770b\u7ed3\u679c"},"\u67e5\u770b\u7ed3\u679c"),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"\u67e5\u770b\u4e32\u53e3\u65e5\u5fd7")),(0,r.kt)("p",null,"CSK6-NanoKit\u901a\u8fc7\u677f\u8f7dDAPlink\u865a\u62df\u4e32\u53e3\u8fde\u63a5\u7535\u8111\uff0c\u6216\u8005\u5c06CSK6-NanoKit\u7684\u65e5\u5fd7\u4e32\u53e3",(0,r.kt)("inlineCode",{parentName:"p"},"A03 TX A02 RX"),"\u5916\u63a5\u4e32\u53e3\u677f\u5e76\u8fde\u63a5\u7535\u8111\u3002"),(0,r.kt)("ul",null,(0,r.kt)("li",{parentName:"ul"},"\u901a\u8fc7lisa\u63d0\u4f9b\u7684",(0,r.kt)("inlineCode",{parentName:"li"},"lisa term"),"\u547d\u4ee4\u67e5\u770b\u65e5\u5fd7"),(0,r.kt)("li",{parentName:"ul"},"\u6216\u8005\u5728\u7535\u8111\u7aef\u4f7f\u7528\u4e32\u53e3\u8c03\u8bd5\u52a9\u624b\u67e5\u770b\u65e5\u5fd7\uff0c\u9ed8\u8ba4\u6ce2\u7279\u7387\u4e3a115200\u3002")),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"slave\u63a5\u6536\u5230master\u53d1\u9001\u7684\u6570\u636e\u7ed3\u679c\u5e94\u4e3a\uff1a")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"data1[0]: 0\ndata1[1]: 1\ndata1[2]: 2\ndata1[3]: 3\n...\ndata1[252]: 252\ndata1[253]: 253\ndata1[254]: 254\ndata1[255]: 0\n")),(0,r.kt)("p",null,(0,r.kt)("strong",{parentName:"p"},"master\u63a5\u6536\u5230slave\u53d1\u9001\u7684\u6570\u636e\u7ed3\u679c\u5e94\u4e3a\uff1a")),(0,r.kt)("pre",null,(0,r.kt)("code",{parentName:"pre"},"slave send data\ndata0[0]: 0\ndata0[1]: 255\ndata0[2]: 254\ndata0[3]: 253\n...\ndata0[252]: 4\ndata0[253]: 3\ndata0[254]: 2\ndata0[255]: 1\n")))}u.isMDXComponent=!0},41612:function(e,n,t){"use strict";n.Z=t.p+"assets/images/i2c_hardware-9cc1f8ba7e2780d9d1a8eb5367193cb1.png"},66960:function(e,n,t){"use strict";n.Z=t.p+"assets/images/i2c_pin_conect-9b9f6410de2d6e97f4fa52936035c975.png"},46425:function(e,n,t){"use strict";n.Z=t.p+"assets/images/lisa_create_01-170e09fdd7c7d4e78bffd6003bbda4ac.png"},3584:function(e,n,t){"use strict";n.Z=t.p+"assets/images/lisa_create_02-eb4d60987bd7b755dd9ad3b6f1262241.png"},2026:function(e,n,t){"use strict";n.Z=t.p+"assets/images/lisa_create_03-acd02cc1a05f2d6ab724e057f98028e9.png"},49682:function(e,n,t){"use strict";n.Z=t.p+"assets/images/lisa_create_04-983eeea45aa26ad08c1f16c5e66ea8f1.png"}}]);