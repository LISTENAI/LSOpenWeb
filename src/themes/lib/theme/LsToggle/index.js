"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.default = _default;

var _react = _interopRequireWildcard(require("react"));

var _ThemedImage = _interopRequireDefault(require("@theme/ThemedImage"));

var _themeCommon = require("@docusaurus/theme-common");

var _useBaseUrl = _interopRequireDefault(require("@docusaurus/useBaseUrl"));

var _useDocusaurusContext = _interopRequireDefault(require("@docusaurus/useDocusaurusContext"));

var _clsx = _interopRequireDefault(require("clsx"));

require("./styles.css");

var _stylesModule = _interopRequireDefault(require("./styles.module.css"));

function _interopRequireDefault(obj) { return obj && obj.__esModule ? obj : { default: obj }; }

function _getRequireWildcardCache(nodeInterop) { if (typeof WeakMap !== "function") return null; var cacheBabelInterop = new WeakMap(); var cacheNodeInterop = new WeakMap(); return (_getRequireWildcardCache = function (nodeInterop) { return nodeInterop ? cacheNodeInterop : cacheBabelInterop; })(nodeInterop); }

function _interopRequireWildcard(obj, nodeInterop) { if (!nodeInterop && obj && obj.__esModule) { return obj; } if (obj === null || typeof obj !== "object" && typeof obj !== "function") { return { default: obj }; } var cache = _getRequireWildcardCache(nodeInterop); if (cache && cache.has(obj)) { return cache.get(obj); } var newObj = {}; var hasPropertyDescriptor = Object.defineProperty && Object.getOwnPropertyDescriptor; for (var key in obj) { if (key !== "default" && Object.prototype.hasOwnProperty.call(obj, key)) { var desc = hasPropertyDescriptor ? Object.getOwnPropertyDescriptor(obj, key) : null; if (desc && (desc.get || desc.set)) { Object.defineProperty(newObj, key, desc); } else { newObj[key] = obj[key]; } } } newObj.default = obj; if (cache) { cache.set(obj, newObj); } return newObj; }

/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
const Dark = ({
  icon,
  style
}) => <span className={(0, _clsx.default)(_stylesModule.default.toggle, _stylesModule.default.dark)} style={style}>
    <_ThemedImage.default sources={icon} />
  </span>;

const Light = ({
  icon,
  style
}) => <span className={(0, _clsx.default)(_stylesModule.default.toggle, _stylesModule.default.light)} style={style}>
    <_ThemedImage.default sources={icon} />
  </span>; // Based on react-toggle (https://github.com/aaronshaf/react-toggle/).


const LsToggle = (0, _react.memo)(({
  className,
  icons,
  checked: defaultChecked,
  disabled,
  onChange
}) => {
  const [checked, setChecked] = (0, _react.useState)(defaultChecked); // const [focused, setFocused] = useState(false);

  const inputRef = (0, _react.useRef)(null);
  return <div className={(0, _clsx.default)('react-toggle', className, {
    'react-toggle--checked': checked,
    // 'react-toggle--focus': focused,
    'react-toggle--disabled': disabled
  })}>
        <div className="react-toggle-track" role="button" tabIndex={-1} onClick={() => {
      var _inputRef$current;

      return (_inputRef$current = inputRef.current) === null || _inputRef$current === void 0 ? void 0 : _inputRef$current.click();
    }}>
          <div className="react-toggle-track-check">{icons.checked}</div>
          <div className="react-toggle-track-x">{icons.unchecked}</div>
          {
        /* <div className="react-toggle-thumb" /> */
      }
        </div>

        <input ref={inputRef} checked={checked} type="checkbox" className="react-toggle-screenreader-only" aria-label="Switch between dark and light mode" onChange={onChange} onClick={() => setChecked(!checked)} // onFocus={() => setFocused(true)}
    // onBlur={() => setFocused(false)}
    />
      </div>;
});

function _default(props) {
  const {
    colorMode: {
      switchConfig: {
        darkIcon,
        darkIconStyle,
        lightIcon,
        lightIconStyle
      }
    }
  } = (0, _themeCommon.useThemeConfig)();
  const {
    isClient
  } = (0, _useDocusaurusContext.default)();
  return <LsToggle disabled={!isClient} icons={{
    checked: <Light icon={{
      light: (0, _useBaseUrl.default)(lightIcon.light),
      dark: (0, _useBaseUrl.default)(lightIcon.dark)
    }} style={lightIconStyle} />,
    unchecked: <Dark icon={{
      light: (0, _useBaseUrl.default)(darkIcon.light),
      dark: (0, _useBaseUrl.default)(darkIcon.dark)
    }} style={darkIconStyle} />
  }} {...props} />;
}