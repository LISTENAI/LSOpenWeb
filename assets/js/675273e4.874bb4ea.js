(self.webpackChunkls_docs_web=self.webpackChunkls_docs_web||[]).push([[5158],{3905:function(e,t,n){"use strict";n.d(t,{Zo:function(){return o},kt:function(){return _}});var r=n(67294);function a(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function l(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);t&&(r=r.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,r)}return n}function i(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?l(Object(n),!0).forEach((function(t){a(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):l(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function p(e,t){if(null==e)return{};var n,r,a=function(e,t){if(null==e)return{};var n,r,a={},l=Object.keys(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||(a[n]=e[n]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(r=0;r<l.length;r++)n=l[r],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(a[n]=e[n])}return a}var u=r.createContext({}),c=function(e){var t=r.useContext(u),n=t;return e&&(n="function"==typeof e?e(t):i(i({},t),e)),n},o=function(e){var t=c(e.components);return r.createElement(u.Provider,{value:t},e.children)},s={inlineCode:"code",wrapper:function(e){var t=e.children;return r.createElement(r.Fragment,{},t)}},d=r.forwardRef((function(e,t){var n=e.components,a=e.mdxType,l=e.originalType,u=e.parentName,o=p(e,["components","mdxType","originalType","parentName"]),d=c(n),_=a,m=d["".concat(u,".").concat(_)]||d[_]||s[_]||l;return n?r.createElement(m,i(i({ref:t},o),{},{components:n})):r.createElement(m,i({ref:t},o))}));function _(e,t){var n=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=n.length,i=new Array(l);i[0]=d;var p={};for(var u in t)hasOwnProperty.call(t,u)&&(p[u]=t[u]);p.originalType=e,p.mdxType="string"==typeof e?e:a,i[1]=p;for(var c=2;c<l;c++)i[c]=n[c];return r.createElement.apply(null,i)}return r.createElement.apply(null,n)}d.displayName="MDXCreateElement"},84421:function(e,t,n){"use strict";n.r(t),n.d(t,{frontMatter:function(){return p},contentTitle:function(){return u},metadata:function(){return c},toc:function(){return o},default:function(){return d}});var r=n(22122),a=n(19756),l=(n(67294),n(3905)),i=["components"],p={},u="UART",c={unversionedId:"chips/600X/application/peripheral/samples/UART",id:"chips/600X/application/peripheral/samples/UART",isDocsHomePage:!1,title:"UART",description:"\u6982\u8ff0",source:"@site/docs/chips/600X/application/peripheral/samples/UART.md",sourceDirName:"chips/600X/application/peripheral/samples",slug:"/chips/600X/application/peripheral/samples/UART",permalink:"/chips/600X/application/peripheral/samples/UART",version:"current",frontMatter:{},sidebar:"docs/chips/600X",previous:{title:"I2C",permalink:"/chips/600X/application/peripheral/samples/I2C"},next:{title:"\u65e5\u5fd7",permalink:"/chips/600X/application/modules/logger"}},o=[{value:"\u6982\u8ff0",id:"\u6982\u8ff0",children:[]},{value:"\u51c6\u5907\u5de5\u4f5c",id:"\u51c6\u5907\u5de5\u4f5c",children:[]},{value:"\u521b\u5efa\u9879\u76ee",id:"\u521b\u5efa\u9879\u76ee",children:[]},{value:"\u4ee3\u7801\u5b9e\u73b0",id:"\u4ee3\u7801\u5b9e\u73b0",children:[{value:"I2C\u8bbe\u5907\u6811\u914d\u7f6e",id:"i2c\u8bbe\u5907\u6811\u914d\u7f6e",children:[]},{value:"\u7ec4\u4ef6\u914d\u7f6e",id:"\u7ec4\u4ef6\u914d\u7f6e",children:[]},{value:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790",id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790",children:[]}]},{value:"\u7f16\u8bd1\u548c\u70e7\u5f55",id:"\u7f16\u8bd1\u548c\u70e7\u5f55",children:[]}],s={toc:o};function d(e){var t=e.components,p=(0,a.Z)(e,i);return(0,l.kt)("wrapper",(0,r.Z)({},s,p,{components:t,mdxType:"MDXLayout"}),(0,l.kt)("h1",{id:"uart"},"UART"),(0,l.kt)("h2",{id:"\u6982\u8ff0"},"\u6982\u8ff0"),(0,l.kt)("p",null,"UART\u662f\u6211\u4eec\u6700\u5e38\u7528\u7684\u5916\u8bbe\u529f\u80fd\u4e4b\u4e00\uff0c\u672c\u7ae0\u8282\u5c06\u901a\u8fc7\u4e24\u4e2a\u793a\u4f8b\u8bb2\u89e3\u5982\u4f55\u4f7f\u7528CSK6 SDK\u7684UART API\u63a5\u53e3\u5b9e\u73b0\u6570\u636e\u7684\u6536\u53d1\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"uart_polling \u8f6e\u5bfb\u65b9\u5f0f\u83b7\u53d6\u4e32\u53e3\u6570\u636e\u3002"),(0,l.kt)("li",{parentName:"ul"},"uart_interrupt \u4e2d\u65ad\u56de\u8c03\u7684\u65b9\u5f0f\u83b7\u53d6\u4e32\u53e3\u6570\u636e\u3002")),(0,l.kt)("p",null,"CSK6 \u82af\u7247\u67094\u7ec4UART\u786c\u4ef6\u5916\u8bbe\u3002",(0,l.kt)("br",{parentName:"p"}),"\n","UART\u9a71\u52a8\u5177\u6709\u5982\u4e0b\u7279\u6027\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u652f\u6301 RTS/CTS \u6d41\u63a7\u3002"),(0,l.kt)("li",{parentName:"ul"},"\u652f\u6301DMA\u4f20\u9001(\u53c2\u6570\u53ef\u914d\u7f6e)\u3002")),(0,l.kt)("h2",{id:"\u51c6\u5907\u5de5\u4f5c"},"\u51c6\u5907\u5de5\u4f5c"),(0,l.kt)("p",null,"\u672c\u793a\u4f8b\u57fa\u4e8e ",(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u5b9e\u73b0\uff0c\u4f7f\u7528\u4e86",(0,l.kt)("inlineCode",{parentName:"p"},"uart0(GPIO_A_03, GPIO_A_06)"),"\u8fd9\u7ec4\u4e32\u53e3\uff0c\u5728\u5f00\u53d1\u4e4b\u524d\u9700\u8981\u505a\u5982\u4e0b\u51c6\u5907\u5de5\u4f5c\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},"\u4f7f\u7528\u675c\u90a6\u7ebf\u5c06\u5f00\u53d1\u677f\u4e0a\u7684uart0\u7684TX\u3001RX\u3001GND\u548c\u4e32\u53e3\u8f6c\u63a5\u677f\u63a5\u5165PC\u3002\n",(0,l.kt)("img",{src:n(5961).Z}))),(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("p",{parentName:"li"},"PC\u7aef\u6253\u5f00\u4e32\u53e3\u8c03\u8bd5\u52a9\u624b\uff0c\u6ce2\u7279\u7387\u8bbe\u7f6e\u4e3a115200\uff0c\u6570\u636e\u4f4d8\uff0c\u65e0\u6821\u9a8c\u4f4d\uff0c\u505c\u6b62\u4f4d1\uff0c\u63a5\u6536\u548c\u53d1\u9001\u6570\u636e\u683c\u5f0f\u90fd\u8bbe\u7f6e\u4e3aHEX\uff0c\u5982\u4e0b\u56fe\u793a\uff1a\n",(0,l.kt)("img",{src:n(20075).Z})))),(0,l.kt)("h2",{id:"\u521b\u5efa\u9879\u76ee"},"\u521b\u5efa\u9879\u76ee"),(0,l.kt)("p",null,"\u901a\u8fc7Lisa\u547d\u4ee4\u521b\u5efa\u9879\u76ee\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep create\n")),(0,l.kt)("p",null,(0,l.kt)("img",{src:n(89072).Z}),"\n",(0,l.kt)("img",{src:n(12392).Z}),"\n",(0,l.kt)("img",{src:n(99884).Z}),"\n\u5206\u522b\u521b\u5efauart_interrupt\u548cuart_polling\u4e24\u4e2asample\u3002"),(0,l.kt)("h2",{id:"\u4ee3\u7801\u5b9e\u73b0"},"\u4ee3\u7801\u5b9e\u73b0"),(0,l.kt)("h3",{id:"i2c\u8bbe\u5907\u6811\u914d\u7f6e"},"I2C\u8bbe\u5907\u6811\u914d\u7f6e"),(0,l.kt)("p",null,(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u63d0\u4f9b\u4e86\u591a\u7ec4UART\u3002\u672c\u793a\u4f8b\u4f7f\u7528",(0,l.kt)("inlineCode",{parentName:"p"},"uart0(GPIO_A_03, GPIO_A_06)"),"\uff0c\u56e0\u6b64\u9700\u8981\u5728\u8bbe\u5907\u6811\u4e2d\u5c06\u8fd9\u7ec4GPIO\u590d\u7528\u4e3aUART0\u5f15\u811a\u529f\u80fd\uff0c\u53ef\u901a\u8fc7",(0,l.kt)("inlineCode",{parentName:"p"},"boad overlay"),"\u7684\u65b9\u5f0f\u5b8c\u6210\uff0c\u5177\u4f53\u5b9e\u73b0\u5982\u4e0b\uff1a"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u5728app\u76ee\u5f55\u4e0b\u7684",(0,l.kt)("inlineCode",{parentName:"li"},"csk6002_9s_nano.overlay"),"\u6587\u4ef6\u5e76\u6dfb\u52a0\u5982\u4e0b\u4e32\u53e3\u914d\u7f6e\uff1a")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"&csk6002_9s_nano_pinctrl{\n    pinctrl_uart0_rx_default: uart0_rx_default{\n        pinctrls = <&pinmuxa 6 2>;              //rx pin\n    };\n    \n    pinctrl_uart0_tx_default: uart0_tx_default{\n        pinctrls = <&pinmuxa 3 2>;              //tx pin\n    };\n};\n")),(0,l.kt)("h3",{id:"\u7ec4\u4ef6\u914d\u7f6e"},"\u7ec4\u4ef6\u914d\u7f6e"),(0,l.kt)("p",null,"\u5728prj.conf\u6587\u4ef6\u4e2d\u6253\u5f00uart\u529f\u80fd\u914d\u7f6e:"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"CONFIG_LOG=y\nCONFIG_UART_INTERRUPT_DRIVEN=y\nCONFIG_UART_CSK6=y\n")),(0,l.kt)("h3",{id:"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790"},"\u5e94\u7528\u903b\u8f91\u5b9e\u73b0\u5206\u6790"),(0,l.kt)("p",null,"\u4e0a\u6587\u63d0\u5230\u6709\u4e24\u79cd\u6570\u636e\u63a5\u6536\u7684\u5b9e\u73b0\u65b9\u5f0f\uff0c\u4e0b\u9762\u6211\u4eec\u6765\u8be6\u7ec6\u4ecb\u7ecd\u4e24\u79cd\u6570\u636e\u63a5\u6536\u65b9\u5f0f\u7684\u5b9e\u73b0\uff1a  "),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},"\u901a\u8fc7\u8f6e\u8bad\u7684\u65b9\u5f0f\u8bfb\u53d6\u4e0a\u4f4d\u673a\u53d1\u9001\u7ed9CSK6\u7684\u6570\u636e\uff0c\u6bcf\u6b21\u8bfb\u4e00\u4e2achar\u7c7b\u578b\u6570\u636e\uff0c\u5e76\u5c06\u63a5\u6536\u5230\u7684\u6570\u636e\u53d1\u9001\u7ed9\u4e0a\u4f4d\u673a\u3002"),(0,l.kt)("li",{parentName:"ul"},"\u901a\u8fc7\u4e2d\u65ad\u56de\u8c03\u7684\u65b9\u5f0f\u8bfb\u53d6\u4e0a\u4f4d\u673a\u53d1\u9001\u7ed9CSK6\u7684\u6570\u636e\uff0c\u5e76\u5c06\u63a5\u6536\u5230\u7684\u6570\u636e\u53d1\u9001\u7ed9\u4e0a\u4f4d\u673a\u3002")),(0,l.kt)("h4",{id:"\u8f6e\u8be2\u65b9\u5f0f\u63a5\u6536\u6570\u636e\uff08uart-polling\uff09"},"\u8f6e\u8be2\u65b9\u5f0f\u63a5\u6536\u6570\u636e\uff08uart polling\uff09"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"API \u63a5\u53e3"),"\n\u8f6e\u8be2\u83b7\u53d6\u6570\u636e\u7684\u65b9\u5f0f\u4e2d\u4e3b\u8981\u7528\u5230\u4ee5\u4e0bUART API\u63a5\u53e3\uff0c",(0,l.kt)("inlineCode",{parentName:"p"},"uart_poll_in"),"\u6bcf\u6b21\u63a5\u6536\u4e00\u4e2achar\uff0c\u5faa\u73af\u63a5\u6536\u76f4\u5230\u6570\u636e\u63a5\u6536\u5b8c\u6210\u3002\u66f4\u591aPolling UART API\u63a5\u53e3\u63cf\u8ff0\u53ef\u4ee5\u5728zephyr\u5b98\u7f51",(0,l.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__uart__polling.html"},"Polling UART API"),"\u4e2d\u770b\u5230\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"/*\u4ece\u4e0a\u4f4d\u673a\u63a5\u6536\u4e00\u4e2achar*/\nuart_poll_in()\n/*\u7ed9\u4e0a\u4f4d\u673a\u53d1\u9001\u4e00\u4e2achar*/\nuart_poll_out()\n")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u8f6e\u8be2\u83b7\u53d6\u6570\u636e")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"/*\u901a\u8fc7uart\u8bbe\u5907\u6811label\u83b7\u53d6nodeid*/\n#define UART0 DT_NODELABEL(uart0)\n\nvoid main(void)\n{\n    /* \u83b7\u53d6uart\u8bbe\u5907\u5b9e\u4f8b */\n    const struct device *uart = DEVICE_DT_GET(UART0);\n\n    while(1){\n        if(uart_poll_in(uart, &rx_char) == 0)   /*\u4ece\u4e0a\u4f4d\u673a\u63a5\u6536\u4e00\u4e2achar*/\n            uart_poll_out(uart, rx_char);       /*\u7ed9\u4e0a\u4f4d\u673a\u53d1\u9001\u4e00\u4e2achar*/\n    }\n    return;\n}\n")),(0,l.kt)("h4",{id:"\u4e2d\u65ad\u56de\u8c03\u65b9\u5f0f\u63a5\u6536\u6570\u636e-\uff08uart-interrupt\uff09"},"\u4e2d\u65ad\u56de\u8c03\u65b9\u5f0f\u63a5\u6536\u6570\u636e \uff08uart interrupt\uff09"),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"API \u63a5\u53e3"),"\n\u672c\u793a\u4f8b\u4e2d\u4e3b\u8981\u7528\u5230\u4e00\u4e0b\u51e0\u4e2aInterrupt UART API\u63a5\u53e3\uff0c\u66f4\u591aAPI\u63a5\u53e3\u63cf\u8ff0\u53ef\u4ee5\u5728zephyr\u5b98\u7f51",(0,l.kt)("a",{parentName:"p",href:"https://docs.zephyrproject.org/latest/doxygen/html/group__uart__interrupt.html"},"Interrupt-driven UART API"),"\u4e2d\u770b\u5230\u3002"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"/*\u4e2d\u65ad\u56de\u8c03\u5904\u7406*/\nuart_rx_callback(const struct device *dev, void *user_data)\n/*\u4e2d\u65ad\u56de\u8c03\u8bbe\u7f6e*/\nvoid uart_irq_callback_user_data_set(const struct device *dev,\n                           uart_irq_callback_user_data_t cb,\n                           void *user_data)\n/*\u8fdb\u5165\u4e2d\u65ad\u5904\u7406\u8c03\u7528*/\nint uart_irq_update(const struct device * dev)\n/*\u67e5\u8be2\u662f\u5426\u6709\u4e2d\u65ad\u6302\u8d77*/\nint uart_irq_is_pending(const struct device * dev)\n/*\u68c0\u67e5 UART RX \u7f13\u51b2\u533a\u662f\u5426\u6709\u63a5\u6536\u5230\u5b57\u7b26*/\nint uart_irq_rx_ready(const struct device * dev)\n/*\u8bfb\u53d6\u5b57\u7b26\u4e32*/\nint uart_fifo_read(const struct device *dev, uint8_t *rx_data,\n                 const int size)\n/*\u53d1\u9001\u5b57\u7b26\u4e32*/\nint uart_fifo_fill(const struct device *dev,\n                 const uint8_t *tx_data,\n                 int size)\n")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u4e32\u53e3\u521d\u59cb\u5316")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"/*\u901a\u8fc7uart\u8bbe\u5907\u6811label\u83b7\u53d6nodeid*/\n#define UART0 DT_NODELABEL(uart0)\n\nvoid main(void)\n{\n    /* \u83b7\u53d6uart\u8bbe\u5907\u5b9e\u4f8b */\n    const struct device *uart = DEVICE_DT_GET(UART0);\n\n    /* \u8bbe\u7f6e\u4e2d\u65ad\u56de\u8c03\u51fd\u6570 */\n    uart_irq_callback_user_data_set(uart, uart_rx_callback, NULL);\n\n    /* \u4f7f\u80fdRX\u4e2d\u65ad\u5904\u7406 */\n    uart_irq_rx_enable(uart);\n}\n")),(0,l.kt)("p",null,(0,l.kt)("strong",{parentName:"p"},"\u4e2d\u65ad\u56de\u8c03\u5904\u7406\uff1a")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre",className:"language-C"},"static void uart_rx_callback(const struct device *dev, void *user_data){\n    /*\u8fdb\u5165\u4e2d\u65ad\u5904\u7406\u8c03\u7528*/\n    uart_irq_update(dev);\n    /*\u67e5\u8be2\u662f\u5426\u6709\u4e2d\u65ad\u6302\u8d77*/                   \n    while (uart_irq_is_pending(dev)) {\n        /*\u68c0\u67e5 UART RX \u7f13\u51b2\u533a\u662f\u5426\u6709\u63a5\u6536\u5230\u5b57\u7b26*/\n        if (uart_irq_rx_ready(dev)) {\n            int recv_len;\n            /*\u8bfb\u53d6\u4e0a\u4f4d\u673a\u53d1\u9001\u5b57\u7b26\u4e32*/\n            recv_len = uart_fifo_read(dev, rx_buffer, 1);\n            /*\u628a\u8bfb\u5230\u7684\u5b57\u7b26\u4e32\u91cd\u65b0\u53d1\u9001\u7ed9\u4e0a\u4f4d\u673a*/\n            uart_fifo_fill(dev, rx_buffer, recv_len);\n        }\n    }\n}\n")),(0,l.kt)("h2",{id:"\u7f16\u8bd1\u548c\u70e7\u5f55"},"\u7f16\u8bd1\u548c\u70e7\u5f55"),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u7f16\u8bd1"),"\n\u5728app\u6839\u76ee\u5f55\u4e0b\u901a\u8fc7\u4e00\u4e0b\u6307\u4ee4\u5b8c\u6210\u7f16\u8bd1\uff1a")),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep build -b csk6002_9s_nano\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u70e7\u5f55"),"   ")),(0,l.kt)("p",null,(0,l.kt)("inlineCode",{parentName:"p"},"csk6002_9s_nano"),"\u5f00\u53d1\u677f\u901a\u8fc7USB\u8fde\u63a5PC\uff0c\u901a\u8fc7\u70e7\u5f55\u6307\u4ee4\u5f00\u59cb\u70e7\u5f55\uff1a"),(0,l.kt)("pre",null,(0,l.kt)("code",{parentName:"pre"},"lisa zep flash --runner pyocd\n")),(0,l.kt)("ul",null,(0,l.kt)("li",{parentName:"ul"},(0,l.kt)("strong",{parentName:"li"},"\u67e5\u770b\u7ed3\u679c"),"  ")),(0,l.kt)("p",null,"\u4e0a\u4f4d\u673a\u7ed9CSK6\u53d1\u9001\u4e00\u4e32\u6307\u4ee4\uff1a",(0,l.kt)("inlineCode",{parentName:"p"},"FF 01 02 03 04 05 06 07 08 09 0A FF"),"\uff0cCSK6\u63a5\u6536\u6307\u4ee4\u5e76\u7ed9\u4e0a\u4f4d\u673a\u56de\u4f20\uff0c\u5982\u4e0b\u56fe\u793a\uff1a\n",(0,l.kt)("img",{src:n(51112).Z})))}d.isMDXComponent=!0},20075:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_01-cbe1c09f4b0e0dad292afb1733d920ee.png"},89072:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_create01-4190be564317cfb86204637eaa81c2cb.png"},12392:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_create02-4925f6a8161d650b7729d9e2b24dd2cd.png"},99884:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_create03-4c5f9f410459175807abc346da63b173.png"},5961:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_pin_conect-f7cb9838f84e2a30cb4bc6e4acd53800.png"},51112:function(e,t,n){"use strict";t.Z=n.p+"assets/images/uart_result-cdf5210b9584517c2a181c823f128f03.png"}}]);